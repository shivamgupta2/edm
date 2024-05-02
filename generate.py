# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Generate random images using the techniques described in the paper
"Elucidating the Design Space of Diffusion-Based Generative Models"."""

import os
import re
import click
import tqdm
import pickle
import numpy as np
import torch
import PIL.Image
import dnnlib
import math
import glob
import random
from torch_utils import distributed as dist
from forwards import Inpainting
import torchvision.transforms.functional as TF
from torch.distributions.multivariate_normal import MultivariateNormal

#----------------------------------------------------------------------------
# Proposed EDM sampler (Algorithm 2).

def edm_sampler(
    net, latents, class_labels=None, randn_like=torch.randn_like,
    num_steps=18, sigma_min=0.002, sigma_max=80, rho=7,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,
):
    # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    # Time step discretization.
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]) # t_N = 0

    # Main sampling loop.
    x_next = latents.to(torch.float64) * t_steps[0]
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
        x_cur = x_next

        # Increase noise temporarily.
        gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
        t_hat = net.round_sigma(t_cur + gamma * t_cur)
        x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur)

        # Euler step.
        denoised = net(x_hat, t_hat, class_labels).to(torch.float64)
        d_cur = (x_hat - denoised) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur

        # Apply 2nd order correction.
        if i < num_steps - 1:
            denoised = net(x_next, t_next, class_labels).to(torch.float64)
            d_prime = (x_next - denoised) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

    return x_next

def twisted_diffusion(measurement_A, measurement_var, y, num_samples, num_particles, num_channels, row_dim, col_dim,
    net, latents, class_labels=None, randn_like=torch.randn_like,
    num_steps=18, sigma_min=None, sigma_max=None, rho=7,
    solver='heun', discretization='edm', schedule='linear', scaling='none',
    epsilon_s=1e-3, C_1=0.001, C_2=0.008, M=1000, alpha=1,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,
):

    assert solver in ['euler', 'heun']
    assert discretization in ['vp', 've', 'iddpm', 'edm']
    assert schedule in ['vp', 've', 'linear']
    assert scaling in ['vp', 'none']

    # Helper functions for VP & VE noise level schedules.
    vp_sigma = lambda beta_d, beta_min: lambda t: (np.e ** (0.5 * beta_d * (t ** 2) + beta_min * t) - 1) ** 0.5
    vp_sigma_deriv = lambda beta_d, beta_min: lambda t: 0.5 * (beta_min + beta_d * t) * (sigma(t) + 1 / sigma(t))
    vp_sigma_inv = lambda beta_d, beta_min: lambda sigma: ((beta_min ** 2 + 2 * beta_d * (sigma ** 2 + 1).log()).sqrt() - beta_min) / beta_d
    ve_sigma = lambda t: t.sqrt()
    ve_sigma_deriv = lambda t: 0.5 / t.sqrt()
    ve_sigma_inv = lambda sigma: sigma ** 2

    # Select default noise level range based on the specified time step discretization.
    if sigma_min is None:
        vp_def = vp_sigma(beta_d=19.9, beta_min=0.1)(t=epsilon_s)
        sigma_min = {'vp': vp_def, 've': 0.02, 'iddpm': 0.002, 'edm': 0.002}[discretization]
    if sigma_max is None:
        vp_def = vp_sigma(beta_d=19.9, beta_min=0.1)(t=1)
        sigma_max = {'vp': vp_def, 've': 100, 'iddpm': 81, 'edm': 80}[discretization]

    # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    # Compute corresponding betas for VP.
    vp_beta_d = 2 * (np.log(sigma_min ** 2 + 1) / epsilon_s - np.log(sigma_max ** 2 + 1)) / (epsilon_s - 1)
    vp_beta_min = np.log(sigma_max ** 2 + 1) - 0.5 * vp_beta_d

    # Define time steps in terms of noise level.
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    if discretization == 'vp':
        orig_t_steps = 1 + step_indices / (num_steps - 1) * (epsilon_s - 1)
        sigma_steps = vp_sigma(vp_beta_d, vp_beta_min)(orig_t_steps)
    elif discretization == 've':
        orig_t_steps = (sigma_max ** 2) * ((sigma_min ** 2 / sigma_max ** 2) ** (step_indices / (num_steps - 1)))
        sigma_steps = ve_sigma(orig_t_steps)
    elif discretization == 'iddpm':
        u = torch.zeros(M + 1, dtype=torch.float64, device=latents.device)
        alpha_bar = lambda j: (0.5 * np.pi * j / M / (C_2 + 1)).sin() ** 2
        for j in torch.arange(M, 0, -1, device=latents.device): # M, ..., 1
            u[j - 1] = ((u[j] ** 2 + 1) / (alpha_bar(j - 1) / alpha_bar(j)).clip(min=C_1) - 1).sqrt()
        u_filtered = u[torch.logical_and(u >= sigma_min, u <= sigma_max)]
        sigma_steps = u_filtered[((len(u_filtered) - 1) / (num_steps - 1) * step_indices).round().to(torch.int64)]
    else:
        assert discretization == 'edm'
        sigma_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho

    # Define noise level schedule.
    if schedule == 'vp':
        sigma = vp_sigma(vp_beta_d, vp_beta_min)
        sigma_deriv = vp_sigma_deriv(vp_beta_d, vp_beta_min)
        sigma_inv = vp_sigma_inv(vp_beta_d, vp_beta_min)
    elif schedule == 've':
        sigma = ve_sigma
        sigma_deriv = ve_sigma_deriv
        sigma_inv = ve_sigma_inv
    else:
        assert schedule == 'linear'
        sigma = lambda t: t
        sigma_deriv = lambda t: 1
        sigma_inv = lambda sigma: sigma

    # Define scaling schedule.
    if scaling == 'vp':
        s = lambda t: 1 / (1 + sigma(t) ** 2).sqrt()
        s_deriv = lambda t: -sigma(t) * sigma_deriv(t) * (s(t) ** 3)
    else:
        assert scaling == 'none'
        s = lambda t: 1
        s_deriv = lambda t: 0

    batch_size = y.shape[0]
    num_y_channels = y.shape[1]
    y_dim = y.shape[2]
    # Compute final time steps based on the corresponding noise levels.
    t_steps = sigma_inv(net.round_sigma(sigma_steps))
    t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])]) # t_N = 0

    t_cur = t_steps[0]
    t_hat = sigma_inv(net.round_sigma(sigma(t_cur)))

    #y = torch.reshape(y, (num_y_channels * row_dim * col_dim,))
    y = torch.reshape(y, (batch_size, num_y_channels * y_dim,))
    #y = torch.flatten(y)
    rev_indices = torch.arange(num_steps -1, -1, -1, device=latents.device).long()
    rev_t_steps = t_steps[rev_indices]

    cond_samples = torch.randn((batch_size, num_samples, num_particles, num_channels * row_dim * col_dim), dtype=torch.double, device=latents.device) * (sigma(t_cur) * s(t_cur))
    #cond_samples = latents.to(torch.float64) * (sigma(t_cur) * s(t_cur))
    #print('cond_samples:', cond_samples)

    cond_samples_shaped = torch.reshape(cond_samples, (batch_size * num_samples * num_particles, num_channels, row_dim, col_dim))
    denoised = net(cond_samples_shaped/s(t_hat), sigma(t_hat), class_labels).to(torch.float64)
    x_0_given_x_T = denoised
    #x_0_given_x_T = (1/s(t_cur)) * (cond_samples_shaped + (s(t_cur)**2) * (denoised - cond_samples_shaped))
    #A_x_0_given_x_T = torch.mul(measurement_A, torch.reshape(x_0_given_x_T, (num_samples * num_particles, num_channels, row_dim * col_dim)))
    A_x_0_given_x_T = measurement_A(torch.reshape(x_0_given_x_T, (batch_size, num_samples * num_particles, num_channels, row_dim * col_dim)))
    A_x_0_given_x_T = torch.reshape(A_x_0_given_x_T, (batch_size, num_samples, num_particles, num_y_channels * y_dim))
    x_0_given_x_T = torch.reshape(x_0_given_x_T, (batch_size, num_samples, num_particles, num_channels * row_dim * col_dim))

    log_Tilde_p_T = vectorized_gaussian_log_pdf(y, A_x_0_given_x_T, measurement_var)

    log_w = log_Tilde_p_T
    
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
        print('it:', i)

        t_hat = sigma_inv(net.round_sigma(sigma(t_cur)))
        step_size = t_hat - t_next

        log_w -= torch.logsumexp(log_w, dim=-1)[:,:,None]
        w = torch.exp(log_w)
    
        resampled_indices = vectorized_random_choice(w, device=latents.device)

        cond_samples = cond_samples[torch.arange(batch_size, device=latents.device)[:, None, None], torch.arange(num_samples, device=latents.device)[None, :,None], resampled_indices]
        log_Tilde_p_T = log_Tilde_p_T[torch.arange(batch_size, device=latents.device)[:, None, None], torch.arange(num_samples, device=latents.device)[None, :,None], resampled_indices]

        cond_samples = cond_samples.requires_grad_()

        x_hat = torch.reshape(cond_samples, (batch_size * num_samples * num_particles, num_channels, row_dim, col_dim))
        denoised = net(x_hat / s(t_hat), sigma(t_hat), class_labels).to(torch.float64)
        x_0_given_x_T = torch.reshape(denoised, (batch_size, num_samples, num_particles, num_channels * row_dim * col_dim))
        #x_0_given_x_T = (1/s(t_hat)) * (cond_samples + (s(t_hat)**2) * (denoised_shaped - cond_samples))
        #A_x_0_given_x_T = torch.mul(measurement_A, torch.reshape(x_0_given_x_T, (num_samples * num_particles, num_y_channels, row_dim * col_dim)))
        A_x_0_given_x_T = measurement_A(torch.reshape(x_0_given_x_T, (batch_size, num_samples * num_particles, num_channels, row_dim * col_dim)))
        A_x_0_given_x_T = torch.reshape(A_x_0_given_x_T, (batch_size, num_samples, num_particles, num_y_channels * y_dim))
        norm_calc = -(1/(2 * measurement_var)) * torch.norm(y[:, None, None, :] - A_x_0_given_x_T, dim=-1)**2
        norm_calc_sum = torch.sum(norm_calc)

        Tilde_p_T_scores = torch.autograd.grad(outputs=norm_calc_sum, inputs=cond_samples)[0]

        A_x_0_given_x_T = A_x_0_given_x_T.detach()
        cond_samples = cond_samples.detach()
        x_0_given_x_T = x_0_given_x_T.detach()
        x_hat = x_hat.detach()
        denoised = denoised.detach()
        Tilde_p_T_scores = Tilde_p_T_scores.detach()

        uncond_score = -((2 * sigma_deriv(t_hat) / sigma(t_hat) + s_deriv(t_hat) / s(t_hat)) * x_hat - 2*sigma_deriv(t_hat) * s(t_hat) / sigma(t_hat) * denoised)
        uncond_score = torch.reshape(uncond_score, (batch_size, num_samples, num_particles, num_channels * row_dim * col_dim))

        cond_score_approx = uncond_score + Tilde_p_T_scores
        #print('Tilde pt scores:', Tilde_p_T_scores)
        cond_samples = torch.reshape(cond_samples, (batch_size, num_samples, num_particles, num_channels * row_dim * col_dim))
        #cond_samples = cond_samples.required_grad_()
        #Tilde_p_T_scores = torch.reshape(Tilde_p_T_scores, (num_samples, num_particles, num_channels * row_dim * col_dim))

        #print('here:', cond_samples.shape, cond_score_approx.shape)
        #print(step_size, randn_like(cond_samples) * torch.sqrt(step_size))
        noise = randn_like(cond_samples) * torch.sqrt(sigma(t_hat) ** 2 - sigma(t_next) ** 2) * s(t_hat)
        next_cond_samples = cond_samples + step_size * cond_score_approx + noise

        next_log_Tilde_p_T = vectorized_gaussian_log_pdf(y, A_x_0_given_x_T, measurement_var)

        log_w_term_1 = vectorized_gaussian_log_pdf(next_cond_samples, cond_samples + step_size * uncond_score, step_size)
        log_w_term_3 = vectorized_gaussian_log_pdf(next_cond_samples, cond_samples + step_size * cond_score_approx, step_size)

        log_w = log_w_term_1 + next_log_Tilde_p_T - log_w_term_3 - log_Tilde_p_T

        cond_samples = next_cond_samples
        log_Tilde_p_T = next_log_Tilde_p_T

        #cond_samples = cond_samples.requires_grad_(False)

        
    cond_samples = cond_samples[:, :,0,:].reshape((batch_size, num_samples, num_channels, row_dim, col_dim))
    return cond_samples

#----------------------------------------------------------------------------
# Generalized ablation sampler, representing the superset of all sampling
# methods discussed in the paper.

def ablation_sampler(
    net, latents, class_labels=None, randn_like=torch.randn_like,
    num_steps=18, sigma_min=None, sigma_max=None, rho=7,
    solver='heun', discretization='edm', schedule='linear', scaling='none',
    epsilon_s=1e-3, C_1=0.001, C_2=0.008, M=1000, alpha=1,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,
):
    assert solver in ['euler', 'heun']
    assert discretization in ['vp', 've', 'iddpm', 'edm']
    assert schedule in ['vp', 've', 'linear']
    assert scaling in ['vp', 'none']

    # Helper functions for VP & VE noise level schedules.
    vp_sigma = lambda beta_d, beta_min: lambda t: (np.e ** (0.5 * beta_d * (t ** 2) + beta_min * t) - 1) ** 0.5
    vp_sigma_deriv = lambda beta_d, beta_min: lambda t: 0.5 * (beta_min + beta_d * t) * (sigma(t) + 1 / sigma(t))
    vp_sigma_inv = lambda beta_d, beta_min: lambda sigma: ((beta_min ** 2 + 2 * beta_d * (sigma ** 2 + 1).log()).sqrt() - beta_min) / beta_d
    ve_sigma = lambda t: t.sqrt()
    ve_sigma_deriv = lambda t: 0.5 / t.sqrt()
    ve_sigma_inv = lambda sigma: sigma ** 2

    # Select default noise level range based on the specified time step discretization.
    if sigma_min is None:
        vp_def = vp_sigma(beta_d=19.9, beta_min=0.1)(t=epsilon_s)
        sigma_min = {'vp': vp_def, 've': 0.02, 'iddpm': 0.002, 'edm': 0.002}[discretization]
    if sigma_max is None:
        vp_def = vp_sigma(beta_d=19.9, beta_min=0.1)(t=1)
        sigma_max = {'vp': vp_def, 've': 100, 'iddpm': 81, 'edm': 80}[discretization]

    # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    # Compute corresponding betas for VP.
    vp_beta_d = 2 * (np.log(sigma_min ** 2 + 1) / epsilon_s - np.log(sigma_max ** 2 + 1)) / (epsilon_s - 1)
    vp_beta_min = np.log(sigma_max ** 2 + 1) - 0.5 * vp_beta_d

    # Define time steps in terms of noise level.
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    if discretization == 'vp':
        orig_t_steps = 1 + step_indices / (num_steps - 1) * (epsilon_s - 1)
        sigma_steps = vp_sigma(vp_beta_d, vp_beta_min)(orig_t_steps)
    elif discretization == 've':
        orig_t_steps = (sigma_max ** 2) * ((sigma_min ** 2 / sigma_max ** 2) ** (step_indices / (num_steps - 1)))
        sigma_steps = ve_sigma(orig_t_steps)
    elif discretization == 'iddpm':
        u = torch.zeros(M + 1, dtype=torch.float64, device=latents.device)
        alpha_bar = lambda j: (0.5 * np.pi * j / M / (C_2 + 1)).sin() ** 2
        for j in torch.arange(M, 0, -1, device=latents.device): # M, ..., 1
            u[j - 1] = ((u[j] ** 2 + 1) / (alpha_bar(j - 1) / alpha_bar(j)).clip(min=C_1) - 1).sqrt()
        u_filtered = u[torch.logical_and(u >= sigma_min, u <= sigma_max)]
        sigma_steps = u_filtered[((len(u_filtered) - 1) / (num_steps - 1) * step_indices).round().to(torch.int64)]
    else:
        assert discretization == 'edm'
        sigma_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho

    # Define noise level schedule.
    if schedule == 'vp':
        sigma = vp_sigma(vp_beta_d, vp_beta_min)
        sigma_deriv = vp_sigma_deriv(vp_beta_d, vp_beta_min)
        sigma_inv = vp_sigma_inv(vp_beta_d, vp_beta_min)
    elif schedule == 've':
        sigma = ve_sigma
        sigma_deriv = ve_sigma_deriv
        sigma_inv = ve_sigma_inv
    else:
        assert schedule == 'linear'
        sigma = lambda t: t
        sigma_deriv = lambda t: 1
        sigma_inv = lambda sigma: sigma

    # Define scaling schedule.
    if scaling == 'vp':
        s = lambda t: 1 / (1 + sigma(t) ** 2).sqrt()
        s_deriv = lambda t: -sigma(t) * sigma_deriv(t) * (s(t) ** 3)
    else:
        assert scaling == 'none'
        s = lambda t: 1
        s_deriv = lambda t: 0

    # Compute final time steps based on the corresponding noise levels.
    t_steps = sigma_inv(net.round_sigma(sigma_steps))
    t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])]) # t_N = 0

    # Main sampling loop.
    t_next = t_steps[0]
    x_next = latents.to(torch.float64) * (sigma(t_next) * s(t_next))
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
        x_cur = x_next

        # Increase noise temporarily.
        gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= sigma(t_cur) <= S_max else 0
        t_hat = sigma_inv(net.round_sigma(sigma(t_cur) + gamma * sigma(t_cur)))
        x_hat = s(t_hat) / s(t_cur) * x_cur + (sigma(t_hat) ** 2 - sigma(t_cur) ** 2).clip(min=0).sqrt() * s(t_hat) * S_noise * randn_like(x_cur)

        # Euler step.
        h = t_next - t_hat
        denoised = net(x_hat / s(t_hat), sigma(t_hat), class_labels).to(torch.float64)
        d_cur = (sigma_deriv(t_hat) / sigma(t_hat) + s_deriv(t_hat) / s(t_hat)) * x_hat - sigma_deriv(t_hat) * s(t_hat) / sigma(t_hat) * denoised
        x_prime = x_hat + alpha * h * d_cur
        t_prime = t_hat + alpha * h

        # Apply 2nd order correction.
        if solver == 'euler' or i == num_steps - 1:
            x_next = x_hat + h * d_cur
        else:
            assert solver == 'heun'
            denoised = net(x_prime / s(t_prime), sigma(t_prime), class_labels).to(torch.float64)
            d_prime = (sigma_deriv(t_prime) / sigma(t_prime) + s_deriv(t_prime) / s(t_prime)) * x_prime - sigma_deriv(t_prime) * s(t_prime) / sigma(t_prime) * denoised
            x_next = x_hat + h * ((1 - 1 / (2 * alpha)) * d_cur + 1 / (2 * alpha) * d_prime)

    return x_next

#vectorized log pdfs for multiple mean vectors (given in shape (num_samples, num_particles, dim)), all with same cov, with multiple x's in shape (num_samples, dim)
#could also be that x has same shape as mean vectors
#returns logpdfs in shape(num_samples, num_particles)
def vectorized_gaussian_log_pdf(x, means, variance):
    #_, d = covariance.shape
    d = len(means)
    constant = d * np.log(2 * np.pi)
    #_, log_det = torch.linalg.slogdet(covariance)
    log_det = d * math.log(float(variance))
    #cov_inv = torch.linalg.inv(covariance)
    if x.shape == means.shape:
        deviations = x - means
    else:
        deviations = x[:, None, None, :] - means
    main_term = (torch.norm(deviations, dim=-1) ** 2)/variance
    logprobs = -0.5 * (constant + log_det + main_term)
    #logprobs = -0.5 * (constant + log_det + torch.einsum('ijk,kl,ijl->ij', deviations, cov_inv, deviations))
    return logprobs

def vectorized_gaussian_score(measurement_A, x, means, var):
    if x.shape == means.shape:
        deviations = - (x - means)
    else:
        deviations = -(x[None, None, :] -means)
    return deviations/var

def vectorized_random_choice(probs, device):
    cumulative_probs = torch.cumsum(probs, dim=-1)[:, :, None, :]
    unif_samples = torch.rand(probs.shape[0], probs.shape[1], probs.shape[2], device=device)
    helper = cumulative_probs > unif_samples[:,:,:,None]
    helper = helper.double()
    res = helper.argmax(dim=-1)
    return res

def particle_filtering_sampler(measurement_A, measurement_var, y, num_samples, num_particles, num_channels, row_dim, col_dim, net, latents, class_labels=None, randn_like=torch.randn_like,
    num_steps=18, sigma_min=None, sigma_max=None, rho=7,
    solver='heun', discretization='edm', schedule='linear', scaling='none',
    epsilon_s=1e-3, C_1=0.001, C_2=0.008, M=1000, alpha=1,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,
):

    assert solver in ['euler', 'heun']
    assert discretization in ['vp', 've', 'iddpm', 'edm']
    assert schedule in ['vp', 've', 'linear']
    assert scaling in ['vp', 'none']

    # Helper functions for VP & VE noise level schedules.
    vp_sigma = lambda beta_d, beta_min: lambda t: (np.e ** (0.5 * beta_d * (t ** 2) + beta_min * t) - 1) ** 0.5
    vp_sigma_deriv = lambda beta_d, beta_min: lambda t: 0.5 * (beta_min + beta_d * t) * (sigma(t) + 1 / sigma(t))
    vp_sigma_inv = lambda beta_d, beta_min: lambda sigma: ((beta_min ** 2 + 2 * beta_d * (sigma ** 2 + 1).log()).sqrt() - beta_min) / beta_d
    ve_sigma = lambda t: t.sqrt()
    ve_sigma_deriv = lambda t: 0.5 / t.sqrt()
    ve_sigma_inv = lambda sigma: sigma ** 2

    # Select default noise level range based on the specified time step discretization.
    if sigma_min is None:
        vp_def = vp_sigma(beta_d=19.9, beta_min=0.1)(t=epsilon_s)
        sigma_min = {'vp': vp_def, 've': 0.02, 'iddpm': 0.002, 'edm': 0.002}[discretization]
    if sigma_max is None:
        vp_def = vp_sigma(beta_d=19.9, beta_min=0.1)(t=1)
        sigma_max = {'vp': vp_def, 've': 100, 'iddpm': 81, 'edm': 80}[discretization]

    # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    # Compute corresponding betas for VP.
    vp_beta_d = 2 * (np.log(sigma_min ** 2 + 1) / epsilon_s - np.log(sigma_max ** 2 + 1)) / (epsilon_s - 1)
    vp_beta_min = np.log(sigma_max ** 2 + 1) - 0.5 * vp_beta_d

    # Define time steps in terms of noise level.
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    if discretization == 'vp':
        orig_t_steps = 1 + step_indices / (num_steps - 1) * (epsilon_s - 1)
        sigma_steps = vp_sigma(vp_beta_d, vp_beta_min)(orig_t_steps)
    elif discretization == 've':
        orig_t_steps = (sigma_max ** 2) * ((sigma_min ** 2 / sigma_max ** 2) ** (step_indices / (num_steps - 1)))
        sigma_steps = ve_sigma(orig_t_steps)
    elif discretization == 'iddpm':
        u = torch.zeros(M + 1, dtype=torch.float64, device=latents.device)
        alpha_bar = lambda j: (0.5 * np.pi * j / M / (C_2 + 1)).sin() ** 2
        for j in torch.arange(M, 0, -1, device=latents.device): # M, ..., 1
            u[j - 1] = ((u[j] ** 2 + 1) / (alpha_bar(j - 1) / alpha_bar(j)).clip(min=C_1) - 1).sqrt()
        u_filtered = u[torch.logical_and(u >= sigma_min, u <= sigma_max)]
        sigma_steps = u_filtered[((len(u_filtered) - 1) / (num_steps - 1) * step_indices).round().to(torch.int64)]
    else:
        assert discretization == 'edm'
        sigma_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho

    # Define noise level schedule.
    if schedule == 'vp':
        sigma = vp_sigma(vp_beta_d, vp_beta_min)
        sigma_deriv = vp_sigma_deriv(vp_beta_d, vp_beta_min)
        sigma_inv = vp_sigma_inv(vp_beta_d, vp_beta_min)
    elif schedule == 've':
        sigma = ve_sigma
        sigma_deriv = ve_sigma_deriv
        sigma_inv = ve_sigma_inv
    else:
        assert schedule == 'linear'
        sigma = lambda t: t
        sigma_deriv = lambda t: 1
        sigma_inv = lambda sigma: sigma

    # Define scaling schedule.
    if scaling == 'vp':
        s = lambda t: 1 / (1 + sigma(t) ** 2).sqrt()
        s_deriv = lambda t: -sigma(t) * sigma_deriv(t) * (s(t) ** 3)
    else:
        assert scaling == 'none'
        s = lambda t: 1
        s_deriv = lambda t: 0

    # Compute final time steps based on the corresponding noise levels.
    t_steps = sigma_inv(net.round_sigma(sigma_steps))
    t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])]) # t_N = 0

    rev_indices = torch.arange(num_steps -1, -1, -1, device=latents.device).long()
    rev_t_steps = t_steps[rev_indices]
    measurement_block_A = torch.block_diag(measurement_A, measurement_A, measurement_A)
    measurement_block_A = measurement_block_A.to(torch.double)

    noise_for_y = torch.zeros((num_samples, num_steps, num_channels * row_dim * col_dim), dtype=torch.double, device=latents.device)
    noise_for_y[:, 1:, :] = torch.cumsum(torch.randn((num_samples, num_steps-1, num_channels*row_dim*col_dim), dtype=torch.double, device=latents.device) * torch.sqrt(rev_t_steps[1:] - rev_t_steps[:-1])[:, None], dim=1)
    measured_noise_for_y = torch.einsum('ij,klj->kli', measurement_block_A, noise_for_y)
    y = torch.reshape(y, (num_channels * row_dim * col_dim,))
    noisy_y = (y + measured_noise_for_y)[:, rev_indices, :]

    #we know p(x_N | y_N) propto p(x_N) * p(y_N | x_N)
    #we also know that p(x_N) \approx N(0, schedule[0] * I_d)
    #and that p(y_N | x_N) = N(A x_N, meas_var * I_d)
    #p(x_N | y_N) is then Gaussian with mean (meas_var * I + schdule[0] * A^T A)^{-1} (schedule[0] * A^T y)
    #and covariance (schedule[0] * meas_var) * (meas_var * I + schedule[0] * A^T A)^{-1}
    x_N_cond_y_N_mean = torch.einsum('ij,kj->ki', torch.linalg.inv(measurement_var * torch.eye(num_channels * row_dim * col_dim, device=latents.device) + t_steps[0] * torch.matmul(measurement_block_A.T, measurement_block_A)), t_steps[0] * torch.einsum('ij,kj->ki', measurement_block_A.T, noisy_y[:, 0, :]))
    x_N_cond_y_N_cov = (t_steps[0] * measurement_var) * torch.linalg.inv(measurement_var * torch.eye(num_channels * row_dim * col_dim, device=latents.device) + t_steps[0] * torch.matmul(measurement_block_A.T, measurement_block_A))
    print(x_N_cond_y_N_cov.type())

    mvn = MultivariateNormal(torch.zeros(num_channels * row_dim * col_dim, dtype=torch.double, device=latents.device), x_N_cond_y_N_cov)
    #cur_samples has shape (num_samples, num_particles, dim)
    cur_samples = mvn.sample((num_samples, num_particles)) + x_N_cond_y_N_mean[:, None, :]


    for it in range(1, num_steps):
        step_size = t_steps[it-1] - t_steps[it]
        relevant_inv = torch.linalg.inv(measurement_var * torch.eye(num_channels * row_dim * col_dim, device=latents.device) + step_size * torch.matmul(measurement_block_A.T, measurement_block_A))

        t_hat = t_steps[it-1]
        x_hat = torch.reshape(cur_samples, (num_samples * num_particles, num_channels, row_dim, col_dim))
        denoised = net(x_hat / s(t_hat), sigma(t_hat), class_labels).to(torch.float64)
        uncond_score = -(sigma_deriv(t_hat) / sigma(t_hat) + s_deriv(t_hat) / s(t_hat)) * x_hat - sigma_deriv(t_hat) * s(t_hat) / sigma(t_hat) * denoised
        uncond_score = torch.reshape(uncond_score, (num_samples, num_particles, num_channels * row_dim * col_dim))

        #we know that x_{k-1} | x_k, y_{k-1} is generated with prob. propto p(x_{k-1} | x_k) \cdot p(y_{k-1} | x_{k-1})
        #We have that p(x_{k-1} | x_k) \prop to N(x_k + step_size * uncond_score, step_size * I_d)
        #We have that p(y_{k-1} | x_{k-1}) \prop to N(A x_{k-1}, meas_var * I_d)
        #generate x_{k-1} | x_k, y_k-1 - it is Gaussian with mean (meas_var * I + step_size * A^T A)^{-1} * (meas_var * (x_k + step_size * uncond_score) + step_size * A^T y)
        #and covariance (step_size * meas_var) * (meas_var * I + step_size * A^T A)^{-1}
        x_N_minus_it_covar = (step_size * measurement_var) * relevant_inv
        
        #x_N_minus_it_means_helper has shape (num_samples, dim)
        x_N_minus_it_means_helper = step_size * torch.einsum('ij,kj->ki', measurement_block_A.T, noisy_y[:, it, :])
        x_N_minus_it_means_helper_2 = measurement_var * (cur_samples + step_size * uncond_score) + x_N_minus_it_means_helper[:, None, :]
        #x_N_minus_it_means has shape (num_samples, num_particles, dim)
        relevant_inv = relevant_inv.to(torch.double)
        #print(relevant_inv.type())
        x_N_minus_it_means = torch.einsum('ij,klj->kli', relevant_inv, x_N_minus_it_means_helper_2)

        mvn = MultivariateNormal(torch.zeros(num_channels * row_dim * col_dim, dtype=torch.double, device=latents.device), x_N_minus_it_covar)
        #next samples has shape (num_samples, num_particles, num_channels, dim) as expected
        next_samples = mvn.sample((num_samples, num_particles)) + x_N_minus_it_means

        #resampling particles
        log_probs = vectorized_gaussian_log_pdf(noisy_y[:,it,:], torch.einsum('ij,klj->kli', measurement_block_A, next_samples), measurement_var * torch.eye(num_channels*row_dim*col_dim, dtype=torch.double, device=latents.device))
        log_probs = log_probs + vectorized_gaussian_log_pdf(next_samples, cur_samples + step_size * uncond_score, step_size * torch.eye(num_channels * row_dim * col_dim, dtype=torch.double, device=latents.device))
        log_probs -= vectorized_gaussian_log_pdf(next_samples, x_N_minus_it_means, x_N_minus_it_covar)

        log_probs -= vectorized_gaussian_log_pdf(noisy_y[:,it-1,:], torch.einsum('ij, klj->kli', measurement_block_A, cur_samples), measurement_var * torch.eye(num_channels * row_dim * col_dim, dtype=torch.double, device=latents.device))

        probs = torch.exp(log_probs)
        probs /= torch.sum(probs, dim=1)[:, None]
        sample_ids = vectorized_random_choice(probs, device=latents.device)
        cur_samples = next_samples[torch.arange(num_samples, device=latents.device)[:, None], sample_ids]

    cond_sample_ids = torch.randint(num_particles, (num_samples,), device=latents.device)
    cond_samples = cur_samples[torch.arange(num_samples, device=latents.device), cond_sample_ids]
    cond_samples = cond_samples.reshape((num_samples, num_channels, row_dim, col_dim))
    return cond_samples


#----------------------------------------------------------------------------
# Wrapper for torch.Generator that allows specifying a different random seed
# for each sample in a minibatch.

class StackedRandomGenerator:
    def __init__(self, device, seeds):
        super().__init__()
        self.generators = [torch.Generator(device).manual_seed(int(seed) % (1 << 32)) for seed in seeds]

    def randn(self, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randn(size[1:], generator=gen, **kwargs) for gen in self.generators])

    def randn_like(self, input):
        return self.randn(input.shape, dtype=input.dtype, layout=input.layout, device=input.device)

    def randint(self, *args, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randint(*args, size=size[1:], generator=gen, **kwargs) for gen in self.generators])

#----------------------------------------------------------------------------
# Parse a comma separated list of numbers or ranges and return a list of ints.
# Example: '1,2,5-10' returns [1, 2, 5, 6, 7, 8, 9, 10]

def parse_int_list(s):
    if isinstance(s, list): return s
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        m = range_re.match(p)
        if m:
            ranges.extend(range(int(m.group(1)), int(m.group(2))+1))
        else:
            ranges.append(int(p))
    return ranges

#----------------------------------------------------------------------------

@click.command()
@click.option('--network', 'network_pkl',  help='Network pickle filename', metavar='PATH|URL',                      type=str, required=True)
@click.option('--outdir',                  help='Where to save the output images', metavar='DIR',                   type=str, required=True)
@click.option('--seeds',                   help='Random seeds (e.g. 1,2,5-10)', metavar='LIST',                     type=parse_int_list, default='0-63', show_default=True)
@click.option('--subdirs',                 help='Create subdirectory for every 1000 seeds',                         is_flag=True)
@click.option('--class', 'class_idx',      help='Class label  [default: random]', metavar='INT',                    type=click.IntRange(min=0), default=None)
@click.option('--batch', 'max_batch_size', help='Maximum batch size', metavar='INT',                                type=click.IntRange(min=1), default=64, show_default=True)

@click.option('--steps', 'num_steps',      help='Number of sampling steps', metavar='INT',                          type=click.IntRange(min=1), default=18, show_default=True)
@click.option('--sigma_min',               help='Lowest noise level  [default: varies]', metavar='FLOAT',           type=click.FloatRange(min=0, min_open=True))
@click.option('--sigma_max',               help='Highest noise level  [default: varies]', metavar='FLOAT',          type=click.FloatRange(min=0, min_open=True))
@click.option('--rho',                     help='Time step exponent', metavar='FLOAT',                              type=click.FloatRange(min=0, min_open=True), default=7, show_default=True)
@click.option('--S_churn', 'S_churn',      help='Stochasticity strength', metavar='FLOAT',                          type=click.FloatRange(min=0), default=0, show_default=True)
@click.option('--S_min', 'S_min',          help='Stoch. min noise level', metavar='FLOAT',                          type=click.FloatRange(min=0), default=0, show_default=True)
@click.option('--S_max', 'S_max',          help='Stoch. max noise level', metavar='FLOAT',                          type=click.FloatRange(min=0), default='inf', show_default=True)
@click.option('--S_noise', 'S_noise',      help='Stoch. noise inflation', metavar='FLOAT',                          type=float, default=1, show_default=True)

@click.option('--solver',                  help='Ablate ODE solver', metavar='euler|heun',                          type=click.Choice(['euler', 'heun']))
@click.option('--disc', 'discretization',  help='Ablate time step discretization {t_i}', metavar='vp|ve|iddpm|edm', type=click.Choice(['vp', 've', 'iddpm', 'edm']))
@click.option('--schedule',                help='Ablate noise schedule sigma(t)', metavar='vp|ve|linear',           type=click.Choice(['vp', 've', 'linear']))
@click.option('--scaling',                 help='Ablate signal scaling s(t)', metavar='vp|none',                    type=click.Choice(['vp', 'none']))

def main(network_pkl, outdir, subdirs, seeds, class_idx, max_batch_size, device=torch.device('cuda:1'), **sampler_kwargs):
    """Generate random images using the techniques described in the paper
    "Elucidating the Design Space of Diffusion-Based Generative Models".

    Examples:

    \b
    # Generate 64 images and save them as out/*.png
    python generate.py --outdir=out --seeds=0-63 --batch=64 \\
        --network=https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-cond-vp.pkl

    \b
    # Generate 1024 images using 2 GPUs
    torchrun --standalone --nproc_per_node=2 generate.py --outdir=out --seeds=0-999 --batch=64 \\
        --network=https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-cond-vp.pkl
    """
    dist.init()
    num_batches = ((len(seeds) - 1) // (max_batch_size * dist.get_world_size()) + 1) * dist.get_world_size()
    all_batches = torch.as_tensor(seeds).tensor_split(num_batches)
    rank_batches = all_batches[dist.get_rank() :: dist.get_world_size()]

    # Rank 0 goes first.
    if dist.get_rank() != 0:
        torch.distributed.barrier()

    # Load network.
    dist.print0(f'Loading network from "{network_pkl}"...')
    with dnnlib.util.open_url(network_pkl, verbose=(dist.get_rank() == 0)) as f:
        net = pickle.load(f)['ema'].to(device)

    # Other ranks follow.
    if dist.get_rank() == 0:
        torch.distributed.barrier()

    # Loop over batches.
    #dist.print0(f'Generating {len(seeds)} images to "{outdir}"...')
    #path = '/work/09563/shivamgupta/ls6/edm/CIFAR-10-images/**/*.jpg'
    path = './CIFAR-10-images/test/cat/0000.jpg'
    num_images = 1000
    particle_count = 5
    batch_size = 100
    noise_frac = 0.9
    measurement_var = 0.1
    #for noise_frac in torch.arange(0.8, 1.01, 0.1):
    #for measurement_var in torch.arange(0.1, 1, 0.1):
    #k = random.choice(glob.glob(path, recursive=True))
    #print(k)
    true_img = PIL.Image.open(path)
    true_image = (TF.to_tensor(true_img) - 0.5) * 2
    true_image = true_image.to(device)
    num_channels = 3
    row_dim = 32
    col_dim = 32

    masks = torch.zeros(1, row_dim * col_dim, dtype=torch.bool, device=device)
    num_zeros = int(noise_frac * (row_dim * col_dim))
    num_ones = (row_dim * col_dim) - num_zeros
    mask = torch.hstack((torch.zeros(num_zeros, dtype=torch.bool, device=device), torch.ones(num_ones, dtype=torch.bool, device=device)))
    mask = mask[torch.randperm(row_dim * col_dim)]
    masks[0] = mask
    
    #A = torch.rand((row_dim, col_dim), device=device) > noise_frac
    inpainting_utils = Inpainting(masks, (1, num_channels, row_dim * col_dim,))
    #A = torch.reshape(A, (row_dim * col_dim,))
    #A[50:400, 50:400] = torch.zeros(122500).reshape((350, 350))
    noise = math.sqrt(measurement_var) * torch.randn((1, num_channels, num_ones), device=device)
    #cur_image = images[0].reshape((num_channels, row_dim * col_dim))
    cur_image = true_image.reshape((1, num_channels, row_dim * col_dim))
    measurements = inpainting_utils.forward(cur_image) + noise
    measurement_adjoint = inpainting_utils.adjoint(measurements, device)
    #measurement_adjoint = torch.reshape(measurement_adjoint, (batch_size, num_channels, row_dim, col_dim))
    measurement_adjoint = torch.reshape(measurement_adjoint, (1, num_channels, row_dim, col_dim))
    adjoint_f = open('adjoint_pickle', 'wb+')
    pickle.dump(measurement_adjoint, adjoint_f)
    for particle_count in range(50, 101, 5):
        batch_size = int(500/particle_count)
        for ind in range(700, num_images, batch_size):
            print('batch_size here:', batch_size)
            #img_batch = torch.zeros((batch_size, num_channels, row_dim, col_dim), device=device)
            #for batch_ind in range(batch_size):
            #    k = random.choice(glob.glob(path, recursive=True))
            #    print(k)
            #    img = PIL.Image.open(k)
            #    cur_image = TF.to_tensor(img)
            #    cur_image = (cur_image - 0.5) * 2
            #    img_batch[batch_ind] = cur_image
            torch.distributed.barrier()
            #batch_seeds = torch.tensor([0, 1], device=device)
            #batch_size = len(batch_seeds)
            #if batch_size == 0:
            #    continue

            # Pick latents and labels.
            #rnd = StackedRandomGenerator(device, batch_seeds)
            #latents = rnd.randn([batch_size, net.img_channels, net.img_resolution, net.img_resolution], device=device)
            class_labels = None
            if net.label_dim:
                class_labels = torch.eye(net.label_dim, device=device)[rnd.randint(net.label_dim, size=[batch_size], device=device)]
            if class_idx is not None:
                class_labels[:, :] = 0
                class_labels[:, class_idx] = 1

            # Generate images.
            sampler_kwargs = {key: value for key, value in sampler_kwargs.items() if value is not None}
            have_ablation_kwargs = any(x in sampler_kwargs for x in ['solver', 'discretization', 'schedule', 'scaling'])
            #sampler_fn = ablation_sampler if have_ablation_kwargs else edm_sampler
            #images = sampler_fn(net, latents, class_labels, randn_like=rnd.randn_like, **sampler_kwargs)

            #image = (image - 128)/(127.5)

            #masks = torch.zeros(batch_size, row_dim * col_dim, dtype=torch.bool, device=device)
            #for batch_ind in range(batch_size):
            #    num_zeros = int(noise_frac * (row_dim * col_dim))
            #    num_ones = (row_dim * col_dim) - num_zeros
            #    mask = torch.hstack((torch.zeros(num_zeros, dtype=torch.bool, device=device), torch.ones(num_ones, dtype=torch.bool, device=device)))
            #    mask = mask[torch.randperm(row_dim * col_dim)]
            #    masks[batch_ind] = mask
            
            #A = torch.rand((row_dim, col_dim), device=device) > noise_frac
            #inpainting_utils = Inpainting(masks, (batch_size, num_channels, row_dim * col_dim,))
            ##A = torch.reshape(A, (row_dim * col_dim,))
            ##A[50:400, 50:400] = torch.zeros(122500).reshape((350, 350))
            #noise = math.sqrt(measurement_var) * torch.randn((batch_size, num_channels, num_ones), device=device)
            ##cur_image = images[0].reshape((num_channels, row_dim * col_dim))
            #cur_images = img_batch.reshape((batch_size, num_channels, row_dim * col_dim))
            #measurements = inpainting_utils.forward(cur_images) + noise

            #measurement_adjoint = inpainting_utils.adjoint(measurements, device)
            #measurement_adjoint = torch.reshape(measurement_adjoint, (batch_size, num_channels, row_dim, col_dim))
            #measurement_adjoint = torch.reshape(measurement_adjoint, (1, num_channels, row_dim, col_dim))

            num_samples = batch_size
            num_particles = particle_count
            cur_out_dir = outdir + '_particle_count=' + str(num_particles) + '_noise_frac=' + str(float(noise_frac)) + '_measurement_var:' + str(float(measurement_var))
            #cur_out_dir = 'test_path'
            cur_out_dir = os.path.join(f'same_measurement_experiments_0.9', cur_out_dir)

            try:
                os.makedirs(cur_out_dir)
            except OSError as e:
                pass
            latents = torch.randn([num_samples, num_particles, net.img_channels, net.img_resolution, net.img_resolution], device=device)

            reconstructions = twisted_diffusion(inpainting_utils.forward, measurement_var, measurements, num_samples, num_particles, num_channels, row_dim, col_dim, net, latents, class_labels, randn_like=torch.randn_like, **sampler_kwargs)

            for batch_ind in range(batch_size):
                images_dict = {}
                images_dict['truth'] = cur_image[0]
                images_dict['measurement_adjoint'] = measurement_adjoint[0]
                images_dict['recon'] = reconstructions[0][batch_ind]
                images_dict['measurement'] = measurements[0]
                #images_dict['truth'] = (img_batch[batch_ind] * 127.5 + 128).clip(0,255).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
                #images_dict['measurement_adjoint'] = (measurement_adjoint[batch_ind] * 127.5 + 128).clip(0, 255).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
                #reconstructions_np = (reconstructions[batch_ind] * 127.5 + 128).clip(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
                #for i, recon_np in enumerate(reconstructions_np):
                    #images_dict['recon'] = recon_np
                images_dict['mask'] = masks[0]
                images_dict['measurement_var'] = measurement_var
                images_dict['noise frac'] = noise_frac
                images_dict['particle count'] = particle_count
                pickle_path = os.path.join(cur_out_dir, f'image_{(ind + batch_ind):06d}_pickle')
                pickle_file = open(pickle_path, 'wb+')
                pickle.dump(images_dict, pickle_file)
                pickle_file.close()

            continue


                #image_dir = os.path.join(cur_out_dir, f'truth')
                #os.makedirs(image_dir, exist_ok=True)
                #image_path = os.path.join(image_dir, f'{ind:06d}.png')
                #PIL.Image.fromarray(image_np, 'RGB').save(image_path)

            #reconstructions = particle_filtering_sampler(A, measurement_var, measurement, 2, 2, num_channels, row_dim, col_dim, net, latents, class_labels, randn_like=rnd.randn_like, **sampler_kwargs)
            reconstructions = twisted_diffusion(inpainting_utils.forward, measurement_var, measurements, num_samples, num_particles, num_channels, row_dim, col_dim, net, latents, class_labels, randn_like=torch.randn_like, **sampler_kwargs)
            measurement_adjoint = inpainting_utils.adjoint(measurements, device)
            measurement_adjoint = torch.reshape(measurement_adjoint, (batch_size, num_channels, row_dim, col_dim))
            #measurement = torch.reshape(measurement, (num_channels, row_dim, col_dim))

            #save measurements
            for batch_ind in range(batch_size):
                measurement_adjoint_np = (measurement_adjoint[batch_ind] * 127.5 + 128).clip(0, 255).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
                measurement_dir = os.path.join(cur_out_dir, f'measurement')
                os.makedirs(measurement_dir, exist_ok=True)
                measurement_path = os.path.join(measurement_dir, f'measurement_{ind:06d}.png')
                PIL.Image.fromarray(measurement_adjoint_np, 'RGB').save(measurement_path)

            for batch_ind in range(batch_size):
                print('going to reconstruct')
                reconstructions_np = (reconstructions[batch_ind] * 127.5 + 128).clip(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
                for i, recon_np in enumerate(reconstructions_np):
                    recon_dir = os.path.join(cur_out_dir, f'recon')
                    os.makedirs(recon_dir, exist_ok=True)
                    recon_path = os.path.join(recon_dir, f'{ind:06d}_{i:06d}.png')
                    PIL.Image.fromarray(recon_np, 'RGB').save(recon_path)

    torch.distributed.barrier()
    dist.print0('Done.')
    exit()

    for batch_seeds in tqdm.tqdm(rank_batches, unit='batch', disable=(dist.get_rank() != 0)):
        torch.distributed.barrier()
        batch_size = len(batch_seeds)
        if batch_size == 0:
            continue

        # Pick latents and labels.
        rnd = StackedRandomGenerator(device, batch_seeds)
        latents = rnd.randn([batch_size, net.img_channels, net.img_resolution, net.img_resolution], device=device)
        class_labels = None
        if net.label_dim:
            class_labels = torch.eye(net.label_dim, device=device)[rnd.randint(net.label_dim, size=[batch_size], device=device)]
        if class_idx is not None:
            class_labels[:, :] = 0
            class_labels[:, class_idx] = 1

        # Generate images.
        sampler_kwargs = {key: value for key, value in sampler_kwargs.items() if value is not None}
        have_ablation_kwargs = any(x in sampler_kwargs for x in ['solver', 'discretization', 'schedule', 'scaling'])
        sampler_fn = ablation_sampler if have_ablation_kwargs else edm_sampler
        images = sampler_fn(net, latents, class_labels, randn_like=rnd.randn_like, **sampler_kwargs)

        print('here:', images.shape)
        num_channels = 3
        row_dim = 32
        col_dim = 32

        A_dim = images.shape[2] * images.shape[3]
        measurement_var = 0.1
        #A = torch.zeros((row_dim * row_dim, col_dim * col_dim), device=device)
        #A = torch.eye(row_dim * col_dim, device=device)
        A = torch.ones(row_dim, col_dim, device=device)
        A[:20, :20] = torch.zeros(20, 20)
        A = torch.reshape(A, (row_dim * col_dim,))
        #A[50:400, 50:400] = torch.zeros(122500).reshape((350, 350))
        noise = math.sqrt(measurement_var) * torch.randn((num_channels, row_dim * col_dim), device=device)
        #cur_image = images[0].reshape((num_channels, row_dim * col_dim))
        cur_image = images[0].reshape((num_channels, row_dim * col_dim))
        cur_image = cur_image.to(torch.float)
        measurement = torch.mul(A, cur_image) + noise
        num_samples = 3
        num_particles = 10
        latents = rnd.randn([num_samples, num_particles, net.img_channels, net.img_resolution, net.img_resolution], device=device)

        #reconstructions = particle_filtering_sampler(A, measurement_var, measurement, 2, 2, num_channels, row_dim, col_dim, net, latents, class_labels, randn_like=rnd.randn_like, **sampler_kwargs)
        reconstructions = twisted_diffusion(A, measurement_var, measurement, num_samples, num_particles, num_channels, row_dim, col_dim, net, latents, class_labels, randn_like=rnd.randn_like, **sampler_kwargs)
        measurement = torch.reshape(measurement, (num_channels, row_dim, col_dim))

        # Save images.
        images_np = (images * 127.5 + 128).clip(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
        for seed, image_np in zip(batch_seeds, images_np):
            image_dir = os.path.join(outdir, f'{seed-seed%1000:06d}') if subdirs else outdir
            os.makedirs(image_dir, exist_ok=True)
            image_path = os.path.join(image_dir, f'{seed:06d}.png')
            if image_np.shape[2] == 1:
                PIL.Image.fromarray(image_np[:, :, 0], 'L').save(image_path)
            else:
                PIL.Image.fromarray(image_np, 'RGB').save(image_path)

        #save measurements
        measurement_np = (measurement * 127.5 + 128).clip(0, 255).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
        measurement_dir = os.path.join(outdir, f'measurement')
        os.makedirs(measurement_dir, exist_ok=True)
        measurement_path = os.path.join(measurement_dir, f'measurement.png')
        PIL.Image.fromarray(measurement_np, 'RGB').save(measurement_path)

        print('going to reconstruct')
        reconstructions_np = (reconstructions * 127.5 + 128).clip(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
        print(measurement_np)
        print(reconstructions_np)
        for i, recon_np in enumerate(reconstructions_np):
            recon_dir = os.path.join(outdir, f'recon')
            os.makedirs(recon_dir, exist_ok=True)
            recon_path = os.path.join(recon_dir, f'{i:06d}.png')
            PIL.Image.fromarray(recon_np, 'RGB').save(recon_path)

    # Done.
    torch.distributed.barrier()
    dist.print0('Done.')

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
