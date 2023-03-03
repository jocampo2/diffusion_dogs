from utils import plot_sample_images, get_device
from model import load_model
import argparse

if __name__ == "__main__":

    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fname", help="File name of generated image", default="plots/sample"
    )
    parser.add_argument("-n", help="Generate n images", type=int)
    args = parser.parse_args()

    # Load the model and generate the images
    device = get_device()
    model = load_model(device)
    if args.n:
        for i in range(args.n):
            plot_sample_images(model, device, fname=args.fname + str(i))
    else:
        plot_sample_images(model, device, fname=args.fname)
