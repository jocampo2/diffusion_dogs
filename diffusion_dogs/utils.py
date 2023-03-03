import torch
import matplotlib.pyplot as plt
from random import shuffle
from torchvision import transforms
import numpy as np
import torch.nn.functional as F

IMG_SIZE = 64
T = 300


def linear_beta_schedule(timesteps, start=0.0001, end=0.02):
    return torch.linspace(start, end, timesteps)


# Define beta schedule
betas = linear_beta_schedule(timesteps=T)

# Pre-calculate different terms for closed form
alphas = 1.0 - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)


def get_index_from_list(vals, t, x_shape):
    """
    Returns a specific index t of a passed list of values vals
    while considering the batch dimension.
    """
    batch_size = t.shape[0]
    out = vals.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


@torch.no_grad()
def sample_timestep(x, t, model):
    """
    Calls the model to predict the noise in the image and returns
    the denoised image.
    Applies noise to this image, if we are not in the last step yet.
    """
    betas_t = get_index_from_list(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
        sqrt_one_minus_alphas_cumprod, t, x.shape
    )
    sqrt_recip_alphas_t = get_index_from_list(sqrt_recip_alphas, t, x.shape)

    # Call model (current image - noise prediction)
    model_mean = sqrt_recip_alphas_t * (
        x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t
    )
    posterior_variance_t = get_index_from_list(posterior_variance, t, x.shape)

    if t == 0:
        return model_mean
    else:
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_variance_t) * noise


def process_im(im):

    im = im.detach().cpu().numpy()
    im = (im + 1) / 2
    im = np.transpose(im[0, ...], [1, 2, 0])
    im = im * 255.0
    im = im.astype(np.uint8)

    return im


def get_image_timesteps(model, device):

    img = torch.randn((1, 3, IMG_SIZE, IMG_SIZE), device=device)
    images = []
    for i in range(0, T)[::-1]:
        t = torch.full((1,), i, device=device, dtype=torch.long)
        img = sample_timestep(img, t, model)
        images.append(process_im(img))
    return images


def plot_sample_images(model, device, n=7, fname="sample_plot"):

    images = get_image_timesteps(model, device)
    images = images[::-1]
    fig, ax = plt.subplots(1, n, figsize=(n * 2, 2))

    for ax_i, im_i in enumerate(range(0, T, 30)[:n]):

        ax[ax_i].imshow(images[im_i])
        ax[ax_i].set_yticks([])
        ax[ax_i].set_xticks([])

    plt.savefig(
        f"plots/{fname}", bbox_inches="tight", facecolor="white", transparent=False
    )
