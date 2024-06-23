import torch
import numpy as np
from skimage.util.shape import view_as_windows


def random_crop(imgs, output_size):
    """
    Vectorized way to do random crop using sliding windows
    and picking out random ones
    :param imgs: batch images with shape (B,H,W,C)
    """
    # batch size
    n = imgs.shape[0]
    img_size = imgs.shape[1]
    crop_max = img_size - output_size
    w1 = np.random.randint(0, crop_max, n)
    h1 = np.random.randint(0, crop_max, n)
    # creates all sliding windows combinations of size (output_size)
    windows = view_as_windows(
        imgs, (1, output_size, output_size, 1))[..., 0,:,:, 0]
    # selects a random window for each batch element
    cropped_imgs = windows[np.arange(n), w1, h1]
    cropped_imgs = np.transpose(cropped_imgs, (0, 2, 3, 1))
    return cropped_imgs


def view_as_windows_cuda(x, window_shape):
    """PyTorch CUDA-enabled implementation of view_as_windows"""
    assert isinstance(window_shape, tuple) and len(window_shape) == len(x.shape), \
        'window_shape must be a tuple with same number of dimensions as x'

    slices = tuple(slice(None, None, st) for st in torch.ones(4).long())
    win_indices_shape = [
        x.size(0),
        x.size(1)-int(window_shape[1]),
        x.size(2)-int(window_shape[2]),
        x.size(3)    
    ]

    new_shape = tuple(list(win_indices_shape) + list(window_shape))
    strides = tuple(list(x[slices].stride()) + list(x.stride()))

    return x.as_strided(new_shape, strides)


def random_crop_cuda(x, size=84, w1=None, h1=None, return_w1_h1=False):
    """Vectorized CUDA implementation of random crop"""
    assert isinstance(x, torch.Tensor) and x.is_cuda, \
        'input must be CUDA tensor'

    n = x.shape[0]
    img_size = x.shape[-1]
    crop_max = img_size - size

    if crop_max <= 0:
        if return_w1_h1:
            return x, None, None
        return x

    if w1 is None:
        w1 = torch.LongTensor(n).random_(0, crop_max)
        h1 = torch.LongTensor(n).random_(0, crop_max)

    windows = view_as_windows_cuda(x, (1, size, size, 1))[..., 0,:,:, 0]
    cropped = windows[torch.arange(n), w1, h1]
    cropped = cropped.permute(0, 2, 3, 1)

    if return_w1_h1:
        return cropped, w1, h1

    return cropped


def random_cutout_numpy(imgs, min_cut=10,max_cut=30):
    """
        args:
        imgs: np.array shape (B,H,W,C)
        min / max cut: int, min / max size of cutout
        returns np.array
    """

    n, h, w, c = imgs.shape
    w1 = np.random.randint(min_cut, max_cut, n)
    h1 = np.random.randint(min_cut, max_cut, n)

    cutouts = np.empty((n, h, w, c), dtype=imgs.dtype)
    for i, (img, w11, h11) in enumerate(zip(imgs, w1, h1)):
        cut_img = img.copy()
        cut_img[h11:h11 + h11, w11:w11 + w11, :] = 0
        cutouts[i] = cut_img
    return cutouts


def random_cutout_torch(imgs, min_cut=10,max_cut=30):
    """
        args:
        imgs: torch.tensor shape (B,H,W,C)
        min / max cut: int, min / max size of cutout
        returns np.array
    """

    n, h, w, c = imgs.shape
    w1 = np.random.randint(min_cut, max_cut, n)
    h1 = np.random.randint(min_cut, max_cut, n)

    cutouts = torch.empty_like(imgs)
    for i, (img, w11, h11) in enumerate(zip(imgs, w1, h1)):
        cut_img = img.clone()
        cut_img[h11:h11 + h11, w11:w11 + w11, :] = 0
        cutouts[i] = cut_img
    return cutouts


def center_crop_image(image, output_size):
    """
    :param image: shape (H,W,C)
    """
    h, w = image.shape[:2]
    new_h, new_w = output_size, output_size

    top = (h - new_h)//2
    left = (w - new_w)//2

    image = image[top:top + new_h, left:left + new_w, :]
    return image


# class RandomShiftsAug(nn.Module):
#     def __init__(self, pad):
#         super().__init__()
#         self.pad = pad

#     def forward(self, x):
#         n, c, h, w = x.size()
#         assert h == w
#         padding = tuple([self.pad] * 4)
#         x = F.pad(x, padding, 'replicate')
#         eps = 1.0 / (h + 2 * self.pad)
#         arange = torch.linspace(-1.0 + eps,
#                                 1.0 - eps,
#                                 h + 2 * self.pad,
#                                 device=x.device,
#                                 dtype=x.dtype)[:h]
#         arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
#         base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
#         base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

#         shift = torch.randint(0,
#                               2 * self.pad + 1,
#                               size=(n, 1, 1, 2),
#                               device=x.device,
#                               dtype=x.dtype)
#         shift *= 2.0 / (h + 2 * self.pad)

#         grid = base_grid + shift
#         return F.grid_sample(x,
#                              grid,
#                              padding_mode='zeros',
#                              align_corners=False)