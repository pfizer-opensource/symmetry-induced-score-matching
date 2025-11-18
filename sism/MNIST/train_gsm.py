import argparse
import os

import torch.nn as nn
import torch
from matplotlib import pyplot as plt
from torch.optim.lr_scheduler import StepLR
from torchvision.transforms import transforms, ToPILImage, ToTensor, InterpolationMode
from torchvision.transforms import functional as F
from tqdm import tqdm
import numpy as np


from sism.mnist.mnist_data import MNISTDataModule
from torch.utils.tensorboard import SummaryWriter
import math


def rotate_transform_generate(angle):
    if angle < 0:
        angle = -angle
    return transforms.RandomAffine(degrees=(-angle, angle))


def rotate_image(image, angle):
    return F.rotate(image, float(angle), interpolation=InterpolationMode.BILINEAR)


def get_diffusion_coefficients(T, kind="linear-time", alpha_clip_min=0.001):
    if kind == "linear-time":
        t = np.linspace(1e-3, 1.0, T + 2)
        alphaz_cumprod = 1.0 - t
        alphaz_cumprod = alphaz_cumprod / alphaz_cumprod[0]
        alphaz = alphaz_cumprod[1:] / alphaz_cumprod[:-1]
        betaz = 1 - alphaz
        betaz = np.clip(betaz, 0.0, 1.0 - alpha_clip_min)
        betaz = torch.from_numpy(betaz).float().squeeze()
    elif kind == "cosine":
        s = 0.008
        steps = T + 2
        x = torch.linspace(0, T, steps)
        alphaz_cumprod = torch.cos(((x / T) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphaz_cumprod = alphaz_cumprod / alphaz_cumprod[0]
        alphaz = alphaz_cumprod[1:] / alphaz_cumprod[:-1]
        alphaz = alphaz.clip(min=alpha_clip_min)
        betaz = 1 - alphaz
        betaz = torch.clip(betaz, 0.0, 1.0 - alpha_clip_min).float()
    elif kind == "ddpm":
        betaz = torch.linspace(0.1 / T, 20 / T, T + 1)
    else:
        raise ValueError(f"Unknown kind: {kind}")
    return betaz


def get_sigma(t, clip_min=1e-2):
    variance = (math.pi / 4.0) * t * (180.0 / math.pi)
    variance_clipped = torch.clamp(variance, min=clip_min)
    return variance_clipped


class ScoreNet(nn.Module):
    def __init__(self, transform=None):
        super(ScoreNet, self).__init__()
        self.transform = transform
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(4, 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.dense_layers = nn.Sequential(
            nn.Linear(3 * 3 * 32 + 1, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x, t):
        if self.transform:
            x = self.transform(x)
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.dense_layers(torch.cat((x, t), dim=1))
        return x


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Training settings")

    parser.add_argument("--use_conv", action="store_true", default=True)
    parser.add_argument("--display_images", action="store_true", default=False)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--time_steps", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--save_dir", type=str, default="experiments/GSM")

    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    log_dir = args.save_dir + "/logs"
    model_dir = args.save_dir + "/models"

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    writer = SummaryWriter(log_dir=log_dir)
    use_conv = args.use_conv
    display_images = args.display_images
    device = torch.device(args.device)

    nepochs = args.num_epochs
    batch_size = args.batch_size
    T = args.time_steps

    if use_conv:
        transform = transforms.Compose([transforms.Normalize((0.1307,), (0.3081,))])
        score_net = ScoreNet(transform=transform).to(device)
    else:
        in_dim = 784
        hidden_dim = 32
        score_net = nn.Sequential(
            nn.Linear(in_dim + 1, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        ).to(device)

    datamodule = MNISTDataModule(affine=False, batch_size=batch_size)
    datamodule.setup()

    dataloader = datamodule.train_dataloader()

    optimizer_grad = torch.optim.Adam(score_net.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer_grad, step_size=10, gamma=0.9)

    epoch_losses = []
    epochs = range(nepochs)
    with tqdm(epochs, unit="epoch") as tepoch:
        for epoch in tepoch:
            epoch_loss = 0.0
            num_batches = len(dataloader)
            for i, batch in enumerate(dataloader):
                t = torch.randint(low=1, high=T + 1, size=(len(batch[0]),))
                images = batch[0]
                temb = (t / T).unsqueeze(-1)
                variance = get_sigma(temb).squeeze()
                optimizer_grad.zero_grad()
                angle_parameter = torch.randn_like(
                    torch.empty(
                        len(
                            batch[0],
                        ),
                        dtype=torch.float,
                    )
                )
                angles = angle_parameter * variance.squeeze()

                img_perturbed = torch.stack(
                    [
                        ToTensor()(rotate_image(ToPILImage()(image), angle))
                        for image, angle in zip(images, angles)
                    ]
                )

                # from matplotlib import pyplot as plt
                # images_to_plot = img_perturbed[:5]
                # original_images = images[:5]
                # rotation_angles = angles[:5]
                #
                # fig, axs = plt.subplots(2, len(images_to_plot), figsize=(
                # 10, 4))
                # for i, (img, original_img, angle) in enumerate(zip(images_to_plot, original_images, rotation_angles)):
                #     axs[0, i].imshow(original_img.squeeze(), cmap='gray')  # Original image
                #     axs[0, i].set_title(f'Original, Angle: {angle:.2f}')
                #     axs[0, i].axis('off')
                #
                #     axs[1, i].imshow(img.squeeze(), cmap='gray')  # Rotated image
                #     axs[1, i].set_title(f'Rotated, Angle: {angle:.2f}')
                #     axs[1, i].axis('off')
                #
                # plt.show()

                if not use_conv:
                    img_perturbed_lin = img_perturbed.view(
                        img_perturbed.size(0), -1
                    ).squeeze()
                    input_data = torch.cat([img_perturbed_lin, temb], dim=1)
                    score = score_net(input_data).squeeze()
                else:
                    score = score_net(img_perturbed, temb).squeeze()

                target = -angle_parameter  # / variance
                # We do not need the sum since the targets and 1-dimensional. This way we were summating over the batch
                loss_scores = (score - target).pow(2)  # .sum(-1)
                loss = loss_scores.mean()
                loss.backward()
                nn.utils.clip_grad_norm_(score_net.parameters(), 10.0)
                optimizer_grad.step()
                writer.add_scalar("Batch loss", loss.item(), epoch * num_batches + i)
                epoch_loss += loss.item()

            epoch_loss /= num_batches
            epoch_losses.append(epoch_loss)
            writer.add_scalar("Epoch loss", epoch_loss, epoch)
            tepoch.set_postfix(loss=epoch_loss)

            if display_images and epoch % 100 == 0:
                num_images_to_display = 5
                selected_images = images[:num_images_to_display]
                selected_angles = angles[:num_images_to_display]
                selected_targets = target[:num_images_to_display]
                selected_scores = score[:num_images_to_display]
                selected_ts = temb.squeeze()[:num_images_to_display]
                selected_perturbed_images = img_perturbed[:num_images_to_display]

                fig, axs = plt.subplots(2, num_images_to_display, figsize=(15, 4))

                for i, (image, perturbed_image, angle, target, ts, sc) in enumerate(
                    zip(
                        selected_images,
                        selected_perturbed_images,
                        selected_angles,
                        selected_targets,
                        selected_ts,
                        selected_scores,
                    )
                ):
                    axs[0, i].imshow(image.squeeze(), cmap="gray")
                    axs[0, i].set_title(f"t={ts:.2f}, Angle: {angle:.2f}")
                    axs[0, i].axis("off")

                    axs[1, i].imshow(perturbed_image.squeeze(), cmap="gray")
                    axs[1, i].set_title(f"Target: {target:.2f}, Score: {sc:.2f}")
                    axs[1, i].axis("off")

                # Increase the vertical space between plots
                plt.subplots_adjust(wspace=1.5)

                plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(epoch_losses)
    plt.title("Training Loss per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.show()

    torch.save(
        score_net.state_dict(),
        os.path.join(model_dir, f"model_epoch_{epoch}_T={T}.ckpt"),
    )

    writer.close()

    print("Training completed and model saved.")
