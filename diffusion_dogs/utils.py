import torch
import matplotlib.pyplot as plt
from random import shuffle
from torchvision import transforms
import numpy as np
import torch.nn.functional as F

IMG_SIZE = 64
T = 300


def linear_beta_schedule(timesteps: int, start: float = 0.0001, end: float = 0.02):
    """Returns:
    tf.Tensor of the variance (beta) of the noise at each time step"""
    return torch.linspace(start, end, timesteps)


def beta_schedule(t: int = T):
    """
    Pre-computes the noise levels and variance schedules at each timestep (see https://arxiv.org/abs/2006.11239).

    Returns:
        Tuple[torch.Tensor] of betas and alphas
    """
    # Define beta schedule
    betas = linear_beta_schedule(timesteps=t)

    # Pre-calculate different terms for closed form
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, axis=0)
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
    sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
    posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)

    return betas, sqrt_one_minus_alphas_cumprod, sqrt_recip_alphas, posterior_variance


def get_index_from_list(vals: torch.Tensor, t: int, x_shape: torch.Size):
    """
    Returns a specific index t of a passed list of values vals
    while considering the batch dimension.

    Returns:
        torch.Tensor
    """
    batch_size = t.shape[0]
    out = vals.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


@torch.no_grad()
def sample_timestep(x: torch.Tensor, t: int, model: torch.nn.Module):
    """
    Calls the model to predict the noise in the image and returns
    the denoised image.
    Applies noise to this image, if we are not in the last step yet.

    Returns:
        torch.Tensor of the denoised image
    """

    (
        betas,
        sqrt_one_minus_alphas_cumprod,
        sqrt_recip_alphas,
        posterior_variance,
    ) = beta_schedule()

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


def process_im(im: torch.Tensor):
    """Post processes output image from the model into a 8 bit image

    Returns:
        torch.Tensor of processed image
    """
    im = im.detach().cpu().numpy()
    im = (im + 1) / 2
    im = np.transpose(im[0, ...], [1, 2, 0])
    im = im * 255.0
    im = im.astype(np.uint8)

    return im


def get_image_timesteps(model: torch.nn.Module, device: str):
    """Applies the denoising model T times on the initial noise image

    Returns:
        List[torch.Tensor] of the denoised images at each timestep
    """
    img = torch.randn((1, 3, IMG_SIZE, IMG_SIZE), device=device)
    images = []
    for i in range(0, T)[::-1]:
        t = torch.full((1,), i, device=device, dtype=torch.long)
        img = sample_timestep(img, t, model)
        images.append(process_im(img))
    return images


def plot_sample_images(
    model: torch.nn.Module,
    device: str,
    n_time: int = 7,
    fname: str = "plots/sample_plot",
):
    """Generates and plots images using the diffusion model"""

    # Plot time steps
    images = get_image_timesteps(model, device)
    images = images[::-1]
    fig, ax = plt.subplots(1, n_time, figsize=(n_time * 2, 2))

    for ax_i, im_i in enumerate(range(0, T, 30)[:n_time]):

        ax[ax_i].imshow(images[im_i])
        ax[ax_i].set_yticks([])
        ax[ax_i].set_xticks([])

    plt.savefig(
        fname + "_timesteps", bbox_inches="tight", facecolor="white", transparent=False
    )

    # Plot final image
    plt.figure()
    plt.imshow(images[0])
    plt.xticks([])
    plt.yticks([])
    plt.savefig(
        fname + "_final", bbox_inches="tight", facecolor="white", transparent=False
    )


def get_device():
    """Get current device being used"""
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    return device
