import argparse
import math
import os

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
from sism.datasets import (
    generate_concentric_circle_dataset,
    generate_mog_2d,
    generate_line_distribution,
)
from sism.synthetic.utils import (
    create_score_network,
    get_diffusion_coefficients,
    save_losses,
    evaluate_sample_quality,
    generate_comparison_report,
    create_overall_summary,
)


class Config:
    """Configuration class for the 2D toy datasets experiment."""

    def __init__(self, save_dir: str = "results/2d"):
        self.device = torch.device("cpu")
        self.T = 100
        self.in_dim = 2
        self.hidden_dim = 32
        self.epochs = 1000
        self.batch_size = 128
        self.learning_rate = 1e-3
        self.normalize_target_by_std = True
        self.num_samples = 1000
        self.results_dir = save_dir
        self.gradient_clip_norm = 10.0


class SO2Prior:
    """Prior distribution for SO(2) group with log-normal radius and normal angle."""

    def __init__(self, std_r: float = 1, std_theta: float = 1):
        """
        Initialize the SO(2) prior.

        Args:
            std_r: Standard deviation for log-radius distribution
            std_theta: Standard deviation for angle distribution
        """
        self.std_r = std_r
        self.std_theta = std_theta
        self.log_r_d = torch.distributions.Normal(0, 1)
        self.theta_d = torch.distributions.Normal(0, 1)

    def sample(self, n: int) -> torch.Tensor:
        """
        Sample n points from the prior distribution.

        Args:
            n: Number of samples to generate

        Returns:
            Tensor of shape (n, 2) with (r, theta) coordinates
        """
        r = torch.exp(self.log_r_d.sample((n,)) * self.std_r)
        theta = self.theta_d.sample((n,)) * self.std_theta
        return torch.stack([r, theta], dim=-1)


def coordinate_transform_backward(input: torch.Tensor) -> torch.Tensor:
    """
    Transform from polar (r, theta) to Cartesian (x, y) coordinates.

    Args:
        input: Tensor of shape (..., 2) with (r, theta) coordinates

    Returns:
        Tensor of shape (..., 2) with (x, y) coordinates
    """
    if input.dim() == 1:
        input = input.unsqueeze(dim=0)
    r, theta = input.chunk(2, dim=1)
    x = r * torch.cos(theta)
    y = r * torch.sin(theta)
    return torch.concat([x, y], dim=1)


def coordinate_transform_forward(input: torch.Tensor) -> torch.Tensor:
    """
    Transform from Cartesian (x, y) to polar (r, theta) coordinates.

    Args:
        input: Tensor of shape (..., 2) with (x, y) coordinates

    Returns:
        Tensor of shape (..., 2) with (r, theta) coordinates
    """
    if input.dim() == 1:
        input = input.unsqueeze(dim=0)
    x, y = input.chunk(2, dim=1)
    r = torch.sqrt(x**2 + y**2)
    theta = torch.atan2(y, x)
    return torch.concat([r, theta], dim=1)


def forward_noising_vp(
    input: torch.Tensor,
    s: torch.Tensor,
    m: torch.Tensor,
    in_ambient: bool = True,
    sqrt=True,
):

    if s.ndim == 1:
        s = s.unsqueeze(1)
    if m.ndim == 1:
        m = m.unsqueeze(1)
    # ambient space (x,y) into de-coupled space (r, theta)
    r, theta = coordinate_transform_forward(input).chunk(2, dim=1)
    r = r.log()
    # noise variables
    n_r, n_t = torch.randn_like(r), torch.randn_like(theta)
    s2 = s.pow(2)
    l_r = -r * s2 + s * n_r
    l_t = -theta * s2 + s * n_t
    if in_ambient:
        b = input.shape[0]
        cos, sin = l_t.cos(), l_t.sin()
        A0 = l_r.exp()
        A1 = torch.concat([cos, -sin, sin, cos], dim=-1).view(b, 2, 2)
        xt = A0 * torch.einsum("bij,bj->bi", [A1, input])
        ## x(t) = prod_i (O_i(lambda_i)) @ x(0)
    else:
        if sqrt:
            m = m.sqrt()
        rt = m * r + s * n_r
        tt = m * theta + s * n_t
        input = torch.concat([rt.exp(), tt], dim=1)
        xt = coordinate_transform_backward(input)
    noise = torch.concat([n_r, n_t], dim=1)
    return xt, noise


def get_dataset(mode: str = "mog"):

    if mode == "mog":
        dataset, _ = generate_mog_2d(nsamples=10_000)
    elif mode == "concentric_circles":
        dataset = generate_concentric_circle_dataset(nsamples=10_000)
    elif mode == "line":
        dataset = generate_line_distribution(nsamples=10_000)
    else:
        raise ValueError(f"Unknown mode {mode}")

    if isinstance(dataset, torch.Tensor):
        x = dataset.clone()
    else:
        x = torch.from_numpy(dataset).float()
    ids = np.arange(len(dataset))
    torch_dataset = x.clone()
    return torch_dataset, x, ids


def train_gsm(
    score_net: nn.Module,
    torch_dataset: torch.Tensor,
    ids: np.ndarray,
    mean_coeff: torch.Tensor,
    std_coeff: torch.Tensor,
    config: Config,
) -> tuple:
    """Train Generalized Score Matching model."""
    optimizer = torch.optim.Adam(score_net.parameters(), lr=config.learning_rate)
    epoch_losses = []

    with tqdm(range(config.epochs), unit="epoch") as tepoch:
        for _ in tepoch:
            epoch_loss = 0.0
            np.random.shuffle(ids)
            split = np.array_split(ids, math.ceil(len(ids) / config.batch_size))
            cnt = 0

            for i, idx in enumerate(split):
                t = torch.randint(
                    low=1, high=config.T + 1, size=(len(idx),), device=config.device
                )
                temb = (t / config.T).unsqueeze(-1)
                signal = mean_coeff[t].unsqueeze(1)
                std = std_coeff[t].unsqueeze(1)

                xin = torch_dataset[idx]
                xt, target = forward_noising_vp(
                    input=xin, m=signal, s=std, in_ambient=True, sqrt=True
                )

                optimizer.zero_grad()
                score = score_net(torch.concat([xt, temb], dim=1))
                target = -(target / std) if config.normalize_target_by_std else -target

                assert score.ndim == 2 and target.ndim == 2
                loss = (score - target).pow(2).sum(-1).mean()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    score_net.parameters(), config.gradient_clip_norm
                )
                optimizer.step()

                epoch_loss += loss.item()
                cnt += 1

            epoch_loss /= cnt
            epoch_losses.append(epoch_loss)
            tepoch.set_postfix(loss=epoch_loss)

    return epoch_losses, score_net


def sample_from_gsm(
    score_net: nn.Module,
    betas: torch.Tensor,
    config: Config,
    divergence_component: bool = True,
) -> torch.Tensor:
    """Sample from Generalized Score Matching model."""
    prior = SO2Prior(std_r=1.0, std_theta=1.0)
    chain = np.arange(config.T)
    r_theta = prior.sample(n=config.num_samples).to(config.device)
    xsampled = coordinate_transform_backward(r_theta)

    # Lie algebra basis for dilation and SO(2)
    identity_matrix = torch.eye(2).unsqueeze(0).to(config.device)
    infinitesimal_rotation_matrix = (
        torch.tensor([[0, -1], [1, 0]], dtype=torch.float32)
        .unsqueeze(0)
        .to(config.device)
    )

    with torch.no_grad():
        for _, t in tqdm(enumerate(reversed(chain)), total=config.T):
            t_ = torch.tensor([t] * config.num_samples)

            beta = betas[t_].unsqueeze(-1)
            temb = (t_ / config.T).unsqueeze(-1)
            score = score_net(torch.concat([xsampled, temb], dim=-1))

            r, theta = coordinate_transform_forward(xsampled).chunk(2, dim=1)

            # drift and score
            dr = (0.5 * r.log() + score[:, 0].unsqueeze(-1)) * torch.einsum(
                "bij, bj -> bi", identity_matrix, xsampled
            )
            dtheta = (0.5 * theta + score[:, 1].unsqueeze(-1)) * torch.einsum(
                "bij, bj -> bi", infinitesimal_rotation_matrix, xsampled
            )

            noise = torch.randn_like(score)
            nr = noise[:, 0].unsqueeze(-1) * torch.einsum(
                "bij, bj -> bi", identity_matrix, xsampled
            )
            ntheta = noise[:, 1].unsqueeze(-1) * torch.einsum(
                "bij, bj -> bi", infinitesimal_rotation_matrix, xsampled
            )

            x_update = xsampled + beta * dr
            x_update = x_update + beta * dtheta

            x_update = (
                x_update
                + beta.sqrt() * nr
                + 0.5 * beta * xsampled
                + (
                    0.5 * beta * xsampled if divergence_component else 0.0
                )  # 2.0 * beta * xsampled
            )
            x_update = x_update + beta.sqrt() * ntheta - 0.5 * beta * xsampled + 0.0
            xsampled = x_update

    return xsampled


def sample_from_fisher(
    score_net: nn.Module,
    betas: torch.Tensor,
    config: Config,
) -> torch.Tensor:
    """Sample from Fisher Information model."""
    chain = np.arange(config.T)
    xsampled = torch.randn(config.num_samples, config.in_dim) * 1.0

    with torch.no_grad():
        for _, t in tqdm(enumerate(reversed(chain)), total=config.T):
            t_ = torch.tensor([t] * config.num_samples)
            beta = betas[t_].unsqueeze(-1)
            temb = (t_ / config.T).unsqueeze(-1)
            score = score_net(torch.concat([xsampled, temb], dim=-1))
            xsampled = xsampled + 0.5 * beta * xsampled + beta * score
            xsampled = xsampled + beta.sqrt() * torch.randn_like(xsampled)
    return xsampled


def train_fisher(
    score_net: nn.Module,
    torch_dataset: torch.Tensor,
    ids: np.ndarray,
    mean_coeff: torch.Tensor,
    std_coeff: torch.Tensor,
    config: Config,
) -> tuple:
    """Train Fisher Information model."""
    optimizer = torch.optim.Adam(score_net.parameters(), lr=config.learning_rate)
    epoch_losses = []

    with tqdm(range(config.epochs), unit="epoch") as tepoch:
        for epoch in tepoch:
            epoch_loss = 0.0
            np.random.shuffle(ids)
            split = np.array_split(ids, math.ceil(len(ids) / config.batch_size))
            cnt = 0

            for i, idx in enumerate(split):
                t = torch.randint(low=1, high=config.T + 1, size=(len(idx),))
                temb = (t / config.T).unsqueeze(-1)
                signal = mean_coeff[t].unsqueeze(1)
                std = std_coeff[t].unsqueeze(1)

                xin = torch_dataset[idx]
                noise = torch.randn_like(xin)
                xt = signal.sqrt() * xin + std * noise
                target = noise

                optimizer.zero_grad()
                score = score_net(torch.concat([xt, temb], dim=1))
                target = -(target / std) if config.normalize_target_by_std else -target

                assert score.ndim == 2 and target.ndim == 2
                loss = (score - target).pow(2).sum(-1).mean()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    score_net.parameters(), config.gradient_clip_norm
                )
                optimizer.step()

                epoch_loss += loss.item()
                cnt += 1

            epoch_loss /= cnt
            epoch_losses.append(epoch_loss)
            tepoch.set_postfix(loss=epoch_loss)

    return epoch_losses, score_net


def plot_and_save_results(
    original_data: torch.Tensor, sampled_data: torch.Tensor, title: str, filename: str
) -> None:
    """Plot and save comparison between original and sampled data."""
    plt.figure(figsize=(8, 6))

    # Plot original data with small circles
    plt.scatter(
        original_data[:, 0],
        original_data[:, 1],
        s=3,
        alpha=0.6,
        label="Original",
        color="blue",
        marker="o",
    )

    # Plot sampled data with x markers
    plt.scatter(
        sampled_data[:, 0],
        sampled_data[:, 1],
        s=3,
        alpha=0.6,
        label=f"Sampled ({title})",
        color="orange",
        marker="x",
    )

    plt.title(f"Data Comparison: Original vs {title}")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.axis("equal")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Results saved to {filename}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train and evaluate 2D toy dataset experiments with GSM and Fisher score matching models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--save_dir",
        type=str,
        default="results/2d",
        help="Directory to save results and plots",
    )

    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["mog", "concentric_circles", "line"],
        choices=["mog", "concentric_circles", "line"],
        help="List of datasets to run experiments on",
    )

    return parser.parse_args()


def run_experiment(dataset_mode: str = "mog", config: Config | None = None) -> None:
    """Run the complete experiment for both GSM and Fisher methods."""
    # Initialize configuration if not provided
    if config is None:
        config = Config()

    # Load dataset
    torch_dataset, x, ids = get_dataset(dataset_mode)
    torch_dataset = torch_dataset.to(config.device)

    # Setup diffusion coefficients
    betas = get_diffusion_coefficients(config.T, kind="cosine").to(config.device)
    alpha = 1.0 - betas
    mean_coeff = alpha.cumprod(dim=0)
    std_coeff = (1.0 - mean_coeff).sqrt()

    print(f"Running experiment on {dataset_mode} dataset")
    print(f"Dataset size: {len(torch_dataset)}")
    print(f"Device: {config.device}")

    # ============ GSM Experiment ============
    print("\n=== Training GSM Model ===")
    gsm_score_net = create_score_network(
        config.in_dim + 1, config.hidden_dim, config.in_dim, config.device
    )

    gsm_losses, gsm_score_net = train_gsm(
        gsm_score_net, torch_dataset, ids, mean_coeff, std_coeff, config
    )

    # Save GSM losses
    save_losses(gsm_losses, "GSM", dataset_mode, config.results_dir)

    print("=== Sampling from GSM Model ===")
    gsm_samples = sample_from_gsm(gsm_score_net, betas, config)

    # Plot and save GSM results
    gsm_plot_dir = os.path.join(config.results_dir, dataset_mode)
    os.makedirs(gsm_plot_dir, exist_ok=True)
    plot_and_save_results(
        torch_dataset.cpu().numpy(),
        gsm_samples.cpu().numpy(),
        "GSM",
        os.path.join(gsm_plot_dir, "GSM_samples.png"),
    )

    # Evaluate GSM sample quality
    gsm_metrics = evaluate_sample_quality(
        torch_dataset.cpu(), gsm_samples.cpu(), "GSM", dataset_mode, config.results_dir
    )

    # ============ Fisher Experiment ============
    print("\n=== Training Fisher Model ===")
    fisher_score_net = create_score_network(
        config.in_dim + 1, config.hidden_dim, config.in_dim, config.device
    )

    fisher_losses, fisher_score_net = train_fisher(
        fisher_score_net, torch_dataset, ids, mean_coeff, std_coeff, config
    )

    # Save Fisher losses
    save_losses(fisher_losses, "Fisher", dataset_mode, config.results_dir)

    print("=== Sampling from Fisher Model ===")
    fisher_samples = sample_from_fisher(fisher_score_net, betas, config)

    # Plot and save Fisher results
    fisher_plot_dir = os.path.join(config.results_dir, dataset_mode)
    os.makedirs(fisher_plot_dir, exist_ok=True)
    plot_and_save_results(
        torch_dataset.cpu().numpy(),
        fisher_samples.cpu().numpy(),
        "Fisher",
        os.path.join(fisher_plot_dir, "Fisher_samples.png"),
    )

    # Evaluate Fisher sample quality
    fisher_metrics = evaluate_sample_quality(
        torch_dataset.cpu(),
        fisher_samples.cpu(),
        "Fisher",
        dataset_mode,
        config.results_dir,
    )

    # Generate comparison report
    generate_comparison_report(
        gsm_metrics, fisher_metrics, dataset_mode, config.results_dir
    )

    print(f"\nExperiment completed for {dataset_mode} dataset!")


if __name__ == "__main__":
    # Parse command line arguments
    args = parse_args()

    # Create results directory if it doesn't exist
    os.makedirs(args.save_dir, exist_ok=True)

    # Initialize config with custom save directory
    config = Config(save_dir=args.save_dir)

    print(f"Results will be saved to: {args.save_dir}")
    # Track successfully processed datasets
    processed_datasets = []
    print(f"Running experiments on datasets = {args.datasets}")

    # Run experiments for specified datasets
    for dataset in args.datasets:
        print(f"\n{'='*50}")
        print(f"Running experiment for {dataset} dataset")
        print(f"{'='*50}")

        try:
            run_experiment(dataset, config)
            processed_datasets.append(dataset)
        except Exception as e:
            print(f"Error running experiment for {dataset}: {e}")
            continue

    # Create overall summary
    create_overall_summary(processed_datasets, args.save_dir)

    print(f"\n{'='*50}")
    print("All experiments completed!")
    print(f"Results saved to: {args.save_dir}")
    print(
        f"Individual dataset results in subdirectories: {', '.join(processed_datasets)}"
    )
    print(f"{'='*50}")
