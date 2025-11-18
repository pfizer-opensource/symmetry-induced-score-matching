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
import sys
from functools import partial

sys.path.append("..")
sys.path.append("../..")
from sism.mnist.mnist_data import MNISTDataModule
from torch.utils.tensorboard import SummaryWriter
import math


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def rotate_transform_generate(angle):
    if angle < 0:
        angle = -angle
    return transforms.RandomAffine(degrees=(-angle, angle))


def rotate_image(image, angle):
    return F.rotate(image, float(angle), interpolation=InterpolationMode.BILINEAR)


def get_scheduler(T, kind="linear"):
    if kind == "linear":
        m_min, m_max = 0.001, 0.999
        m_t = np.linspace(m_min, m_max, T)
    elif kind == "sin":
        m_t = 1.0075 ** np.linspace(0, T, T)
        m_t = m_t / m_t[-1]
        m_t[-1] = 0.999

    return m_t


def get_sigma(t, clip_min=1e-2):
    variance = (math.pi / 4.0) * t * (180.0 / math.pi)
    variance_clipped = torch.clamp(variance, min=clip_min)
    return variance_clipped


def get_target(img_0, img_1, m_t, sigma_t, epsilon, kind="grad"):
    assert kind in ["grad", "noise", "x0"]
    if kind == "grad":
        target = m_t * (img_1 - img_0) + sigma_t * epsilon
    elif kind == "noise":
        target = epsilon
    elif kind == "x0":
        target = img_0
    return target


def get_mean_reverse(
    img_t,
    img_0,
    img_1,
    var_t_m1,
    var_t,
    m_t,
    m_t_m1,
):

    f = 1.0 - m_t
    g = 1.0 - m_t_m1

    a_t = (var_t_m1 * f) / (var_t * g)

    v_t_t1 = var_t - var_t_m1 * ((f / g).pow(2))
    a_0 = g * (v_t_t1 / var_t)
    a_1 = m_t_m1 - m_t * a_t
    mu_t = a_t * img_t + a_0 * img_0 + a_1 * img_1
    return mu_t, {"a_t": a_t[0], "a_0": a_0[0], "a_1": a_1[0]}


class ScoreNet(nn.Module):
    def __init__(self, transform=None):
        super(ScoreNet, self).__init__()
        self.transform = transform
        self.down_sample_layers = nn.Sequential(
            nn.Conv2d(2, 4, kernel_size=3, stride=1, padding=1),
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

        self.up_sample_layers = nn.Sequential(
            nn.ConvTranspose2d(
                32, 16, kernel_size=4, stride=2, padding=1, output_padding=1
            ),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(
                16, 8, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.ConvTranspose2d(
                8, 4, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.Conv2d(4, 1, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x, t):
        if self.transform:
            x = self.transform(x)
        t = t.unsqueeze(-1).unsqueeze(-1)
        t = t * torch.ones_like(x)
        x = torch.concat((x, t), dim=1)
        x = self.down_sample_layers(x)
        x = self.up_sample_layers(x)
        return x


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Training settings")

    parser.add_argument("--use_conv", action="store_true", default=False)
    parser.add_argument("--display_images", action="store_true", default=False)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--time_steps", type=int, default=1000)
    parser.add_argument(
        "--scheduler", type=str, default="sin", choices=["linear", "sin"]
    )
    parser.add_argument(
        "--kind", type=str, default="x0", choices=["grad", "noise", "x0"]
    )
    parser.add_argument("--max_variance", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--save_dir", type=str, default="experiments/BBDM")
    parser.add_argument("--id", type=int, default=0)

    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    log_dir = args.save_dir + "/logs"
    model_dir = args.save_dir + "/models"

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    writer = SummaryWriter(log_dir=log_dir + f"/run_{args.id}")
    use_conv = args.use_conv
    display_images = args.display_images
    device = torch.device(args.device)

    nepochs = args.num_epochs
    batch_size = args.batch_size
    T = args.time_steps
    context = True

    if use_conv:
        transform = transforms.Compose([transforms.Normalize((0.1307,), (0.3081,))])
        score_net = ScoreNet(transform=transform).to(device)
    else:
        in_dim = 784
        hidden_dim = 128
        score_net = nn.Sequential(
            nn.Linear((1 + int(context)) * in_dim + 1, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim * 1),
            nn.BatchNorm1d(hidden_dim * 1),
            nn.SiLU(),
            nn.Linear(hidden_dim * 1, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(2 * hidden_dim, in_dim),
        ).to(device)

    datamodule = MNISTDataModule(affine=False, batch_size=batch_size)
    datamodule.setup()

    dataloader = datamodule.train_dataloader()

    optimizer_grad = torch.optim.Adam(score_net.parameters(), lr=args.lr)

    epoch_losses = []
    epochs = range(nepochs)

    to_torch = partial(torch.tensor, dtype=torch.float32, device=device)

    m_t = get_scheduler(T=T + 1, kind=args.scheduler)
    m_tminus = np.append(0, m_t[:-1])

    variance_t = 2.0 * (m_t - m_t**2) * args.max_variance
    variance_tminus = np.append(0.0, variance_t[:-1])
    # delta t|t-1
    variance_t_tminus = (
        variance_t - variance_tminus * ((1.0 - m_t) / (1.0 - m_tminus)) ** 2
    )
    # delta t tilde
    posterior_variance_t = variance_t_tminus * variance_tminus / variance_t

    m_t = to_torch(m_t)
    m_tminus = to_torch(m_tminus)
    variance_t = to_torch(variance_t)
    sigma_t = variance_t.sqrt()
    variance_tminus = to_torch(variance_tminus)
    posterior_variance_t = to_torch(posterior_variance_t)

    train_visualize = False

    with tqdm(epochs, unit="epoch") as tepoch:
        for epoch in tepoch:
            epoch_loss = 0.0
            num_batches = len(dataloader)
            for i, batch in enumerate(dataloader):
                t = torch.randint(low=1, high=T + 1, size=(len(batch[0]),))
                # t = (torch.ones_like(t) * 1).long()
                img_0 = batch[0]
                temb = (t / T).unsqueeze(-1)
                # get fully-augmented images at prior state T
                variance = get_sigma(torch.ones_like(temb)).squeeze()
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

                img_1 = torch.stack(
                    [
                        ToTensor()(rotate_image(ToPILImage()(image), angle))
                        for image, angle in zip(img_0, angles)
                    ]
                )

                # noising as in standard diffusion
                # m_t_ = m_t[t].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                m_t_ = extract(m_t, t, img_0.shape)
                # sigma_t_ = sigma_t[t].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                sigma_t_ = extract(sigma_t, t, img_0.shape)
                epsilon = torch.randn_like(img_0)
                img_t = (1.0 - m_t_) * img_0 + m_t_ * img_1 + sigma_t_ * epsilon

                # (b, 1, 28, 28) input
                if train_visualize:
                    from matplotlib import pyplot as plt

                    intermediate_images = img_t[:5]
                    original_images = img_0[:5]
                    terminal_images = img_1[:5]

                    fig, axs = plt.subplots(
                        3, len(intermediate_images), figsize=(16, 8)
                    )
                    for i, (xt, x0, x1, angle, time, std) in enumerate(
                        zip(
                            intermediate_images,
                            original_images,
                            terminal_images,
                            angles,
                            temb,
                            sigma_t_,
                        )
                    ):

                        std = std.squeeze()
                        time = time.squeeze()
                        axs[0, i].imshow(x0.squeeze(), cmap="gray")  # Original image
                        axs[0, i].set_title(f"x(0): angle={0.0}")
                        axs[0, i].axis("off")

                        axs[1, i].imshow(xt.squeeze(), cmap="gray")  # Diffused image
                        axs[1, i].set_title(f"x(t): t={time:.2f}")
                        axs[1, i].axis("off")

                        axs[2, i].imshow(x1.squeeze(), cmap="gray")  # Rotated image
                        axs[2, i].set_title(f"x(1): angle={angle:.2f}")
                        axs[2, i].axis("off")

                    plt.show()

                if not use_conv:
                    if context:
                        input = torch.cat(
                            (
                                img_t.view(img_t.size(0), -1).squeeze(),
                                img_1.view(img_1.size(0), -1).squeeze(),
                            ),
                            dim=-1,
                        )
                    else:
                        input = img_t.view(img_t.size(0), -1).squeeze()
                    input_data = torch.cat([input, temb], dim=1)
                    score = score_net(input_data)
                    score = score.reshape(score.size(0), 1, 28, 28)
                else:

                    score = score_net(img_t, temb)

                target = get_target(
                    img_0=img_0,
                    img_1=img_1,
                    m_t=m_t_,
                    sigma_t=sigma_t_,
                    epsilon=epsilon,
                    kind=args.kind,
                )
                assert score.shape == target.shape
                if args.kind != "x0":
                    loss_scores = (score - target).pow(2).mean((-1, -2))
                else:
                    loss_scores = nn.BCEWithLogitsLoss(
                        weight=None, size_average=None, reduce=None, reduction="none"
                    )(score, target).mean((-1, -2))

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

    plt.figure(figsize=(10, 5))
    plt.plot(epoch_losses)
    plt.title("Training Loss per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.show()

    torch.save(
        score_net.state_dict(),
        os.path.join(
            model_dir, f"model_id_{args.id}_{args.scheduler}_epoch_{epoch}_T={T}.ckpt"
        ),
    )

    writer.close()

    print("Training completed and model saved.")
