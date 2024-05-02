import torch


class Inpainting:
    def __init__(self, masks, x_shape):
        self.masks = masks
        self.x_shape = x_shape

    def forward(self, x):
        batch_size = x.shape[0]
        dim = x.shape[-1]
        factor = int(torch.numel(x)/(batch_size * dim))
        rep_masks = self.masks.repeat_interleave(factor, dim=0)
        temp_y = torch.reshape(x, (batch_size * factor, dim))[rep_masks]
        y_shape = list(x.shape)
        y_shape[-1] = int(temp_y.shape[-1]/(factor * batch_size))
        return torch.reshape(temp_y, y_shape)

    def adjoint(self, y, device):
        print('y shape:', y.shape)
        flattened_y = torch.flatten(y)
        res = torch.zeros(self.x_shape, device=device)
        batch_size = res.shape[0]
        for batch_ind in range(batch_size):
            print('here:', res[batch_ind][:, self.masks[batch_ind]].shape, self.masks.shape)
            res[batch_ind][:, self.masks[batch_ind]] = y[batch_ind]
        return res

class SuperResolution:
    def __init__(self, factor, x_shape):
        self.factor = factor
        self.x_shape = x_shape

    def forward(self, x):
        batch_size = x.shape[0]
        num_channels = x.shape[1]
        dim = x.shape[-1]
        new_dim = dim//self.factor
        helper_x = torch.reshape(batch_size, num_channels, new_dim, self.factor, new_dim, self.factor)
        res = torch.mean(helper_x, dim=(2, 4))
        return res

