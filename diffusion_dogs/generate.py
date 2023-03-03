import torch
from utils import plot_sample_images
from model import SimpleUnet

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

model = SimpleUnet()
model.load_state_dict(
    torch.load("weights/diffusion_model.pt", map_location=torch.device(device))
)
model.to(device)
plot_sample_images(model, device)
