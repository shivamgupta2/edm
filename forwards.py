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
        batch_size = x_shape[0]
        num_channels = x_shape[1]
        dim = x_shape[-1]
        new_dim = dim//self.factor
        self.helper_shape = (batch_size, num_channels, new_dim, self.factor, new_dim, self.factor)

    #def forward(self, x):
    #    print(x.shape)
    #    batch_size = x.shape[0]
    #    num_channels = x.shape[1]
    #    dim = x.shape[-1]
    #    new_dim = dim//self.factor
    #    cur_helper_shape = (batch_size, num_channels, new_dim, self.factor, new_dim, self.factor)
    #    helper_x = torch.reshape(x, cur_helper_shape)
    #    res = torch.mean(helper_x, dim=(3, 5))
    #    return res

    def forward(self, x):
        print(f"Input shape: {x.shape}")

        # Get the shape of the input tensor
        original_shape = x.shape

        # Assume the last three dimensions are (channels, rows, cols)
        num_channels = original_shape[-3]
        rows = original_shape[-2]
        cols = original_shape[-1]

        # Compute the new dimensions
        new_rows = rows // self.factor
        new_cols = cols // self.factor

        # Reshape the tensor to (..., num_channels, new_rows, factor, new_cols, factor)
        new_shape = original_shape[:-3] + (num_channels, new_rows, self.factor, new_cols, self.factor)
        helper_x = torch.reshape(x, new_shape)

        # Compute the mean along the dimensions (factor, factor)
        res = torch.mean(helper_x, dim=(-1, -3))

        return res

    
    def adjoint(self, y):
        print('xshape:', self.x_shape)
        helper_shape = y.shape[:-3] + (3, 32, 1, 32, 1)
        x_shape = y.shape[:-3] + (3, 32, 32)
        y = torch.unsqueeze(y, -2)
        y = torch.unsqueeze(y, -1)
        #y = torch.unsqueeze(y, 3)
        #y = torch.unsqueeze(y, -1)
        y = y.expand(helper_shape)/(32 * 32)
        #y = y.expand(helper_shape)
        return torch.reshape(y, x_shape)
    
    def get_avg_values(self, x):
        return torch.mean(dim=(2, 3))
