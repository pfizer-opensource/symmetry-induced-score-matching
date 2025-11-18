from functools import partial
import math
import sys
import argparse
import os
from typing import Dict, Tuple, Optional

sys.path.append("../..")
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torchvision.transforms import transforms, ToPILImage, ToTensor
from tqdm import tqdm
import numpy as np


from sism.mnist.train_bridge import (
    rotate_image,
    ScoreNet,
    get_scheduler,
    extract,
    get_mean_reverse,
)
from sism.mnist.mnist_data import MNISTDataModule


class Config:
    """Configuration class for bridge diffusion parameters."""

    def __init__(
        self,
        model_path: str = "experiments/BBDM/models/model_id_0_sin_epoch_99_T=1000.ckpt",
        use_conv: bool = False,
        device: str = "cpu",
        context: bool = True,
        max_variance: float = 1.0,
        T: int = 1000,
        seed: int = 5,
        scheduler_kind: str = "sin",
        stop_t: float = 1.0,
    ):
        self.model_path = model_path
        self.use_conv = use_conv
        self.device = device
        self.context = context
        self.max_variance = max_variance
        self.T = T
        self.seed = seed
        self.scheduler_kind = scheduler_kind
        self.stop_t = stop_t
        self.final_discrete_stop_time = math.floor(stop_t * T)


class MNISTTester:
    """Main class for testing MNIST bridge diffusion models."""

    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device(config.device)
        self.score_net = None
        self.datamodule = None
        self._setup_random_seeds()
        self._setup_scheduler()

    def _setup_random_seeds(self):
        """Set random seeds for reproducibility."""
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)

    def _setup_scheduler(self):
        """Initialize scheduler and variance parameters."""
        to_torch = partial(torch.tensor, dtype=torch.float32, device=self.device)

        # Get scheduler
        m_t = get_scheduler(T=self.config.T + 1, kind=self.config.scheduler_kind)
        m_tminus = np.append(0, m_t[:-1])

        # Calculate variances
        variance_t = 2.0 * (m_t - m_t**2) * self.config.max_variance
        variance_tminus = np.append(0.0, variance_t[:-1])
        variance_t_tminus = (
            variance_t - variance_tminus * ((1.0 - m_t) / (1.0 - m_tminus)) ** 2
        )
        posterior_variance_t = variance_t_tminus * variance_tminus / variance_t

        # Convert to tensors
        self.m_t = to_torch(m_t)
        self.m_tminus = to_torch(m_tminus)
        self.variance_t = to_torch(variance_t)
        self.sigma_t = self.variance_t.sqrt()
        self.variance_tminus = to_torch(variance_tminus)
        self.posterior_variance_t = to_torch(posterior_variance_t)

    def load_model(self) -> nn.Module:
        """Load the bridge diffusion model."""
        if self.config.use_conv:
            transform = transforms.Compose([transforms.Normalize((0.1307,), (0.3081,))])
            score_net = ScoreNet(transform=transform).to(self.device)
        else:
            in_dim = 784
            hidden_dim = 128
            score_net = nn.Sequential(
                nn.Linear((1 + int(self.config.context)) * in_dim + 1, hidden_dim * 2),
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
            ).to(self.device)

        # Load model weights
        if os.path.exists(self.config.model_path):
            score_net.load_state_dict(
                torch.load(self.config.model_path, map_location=self.device)
            )
            print(f"Model loaded from: {self.config.model_path}")
        else:
            print(f"Warning: Model path {self.config.model_path} does not exist!")
            raise ValueError

        score_net.eval()
        self.score_net = score_net
        return score_net

    def setup_data(self) -> MNISTDataModule:
        """Setup MNIST data module."""
        self.datamodule = MNISTDataModule(affine=False)
        self.datamodule.setup()
        return self.datamodule

    def generate_test_images(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate test images with rotations."""
        if self.datamodule is None:
            self.setup_data()

        # Organize data by class
        class_ids = {i: [] for i in range(10)}
        for data in self.datamodule.val_dataloader().dataset:
            class_ids[data[1]].append(data[0])

        # Select one image per class
        img_0 = torch.stack([class_ids[i][self.config.seed] for i in range(10)])

        # Generate random rotation angles
        angle_parameter = torch.randn_like(torch.empty(10, dtype=torch.float))
        lambda_T = angle_parameter * (180 / 2)
        lambda_T = torch.where(
            lambda_T.abs() < 180 / 8, torch.ones_like(lambda_T) * 180 / 8, lambda_T
        )

        # Apply rotations
        img_1 = torch.stack(
            [
                ToTensor()(rotate_image(ToPILImage()(image), angle))
                for image, angle in zip(img_0, lambda_T)
            ]
        ).squeeze()

        return img_0, img_1, lambda_T

    def sample_bridge(
        self,
        img_1: torch.Tensor,
        img_0: torch.Tensor,
        verbose: bool = True,
        save_trajectory: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Sample from the bridge diffusion model."""
        if self.score_net is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        img_t = img_1.clone()
        eps = 1e-6
        num_samples = len(img_1)
        chain = np.arange(self.config.T)

        if self.config.context:
            img_1_ = img_1.view(img_1.size(0), -1).squeeze()

        iterator = enumerate(reversed(chain))
        if verbose:
            iterator = tqdm(iterator, total=self.config.T, desc="Sampling")

        cnt = 0
        trajectory = []
        if save_trajectory:
            trajectory.append(img_t.cpu().numpy())
        with torch.no_grad():
            for i, t in iterator:
                t_tensor = torch.tensor([t] * num_samples)
                n_t = t_tensor + 1
                temb = (t_tensor / self.config.T).unsqueeze(-1)
                temb = temb.clamp(min=eps, max=1.0 - eps)

                # Prepare input
                input_tensor = img_t.view(img_t.size(0), -1).squeeze()
                if self.config.context:
                    input_tensor = torch.concat((input_tensor, img_1_), dim=-1)
                input_data = torch.cat([input_tensor, temb], dim=1)

                # Get score and reconstruct
                score = self.score_net(input_data)
                score = score.reshape(score.size(0), 28, 28)
                img0_recon = score.sigmoid()

                if i <= self.config.T - 2:
                    # Calculate reverse diffusion step
                    m_t_ = extract(self.m_t, t_tensor, img_t.shape)
                    m_nt_ = extract(self.m_t, n_t, img_t.shape)
                    var_t_ = extract(self.variance_t, t_tensor, img_t.shape)
                    var_nt_ = extract(self.variance_t, n_t, img_t.shape)

                    sigma2_t_ = (
                        (var_t_ - var_nt_ * (1.0 - m_t_) ** 2 / (1.0 - m_nt_) ** 2)
                        * var_nt_
                        / var_t_
                    )
                    sigma_t_ = torch.sqrt(sigma2_t_.clamp_min(1e-4))
                    noise = torch.randn_like(img_t)

                    img_tminus_mean, _ = get_mean_reverse(
                        img_t=img_t,
                        img_0=img0_recon,
                        img_1=img_1,
                        m_t_m1=m_t_,
                        m_t=m_nt_,
                        var_t_m1=var_t_,
                        var_t=var_nt_,
                    )
                    img_t = img_tminus_mean + sigma_t_ * noise
                else:
                    img_t = img0_recon

                if save_trajectory:
                    trajectory.append(img_t.cpu().numpy())

                cnt += 1
                if cnt > self.config.final_discrete_stop_time:
                    if verbose:
                        print(f"Stopping early at step {cnt}")
                    break

        return {
            "img": img_t,
            "img_0": img_0,
            "img_1": img_1,
            "trajectory": np.stack(trajectory) if save_trajectory else None,
        }

    def visualize_results(
        self, results: Dict[str, torch.Tensor], save_path: Optional[str] = None
    ):
        """Visualize reconstruction results."""
        img_0 = results["img_0"].squeeze()
        img_1 = results["img_1"].squeeze()
        img_reconstructed = results["img"].squeeze()

        num_samples = min(10, len(img_0))

        # Convert to numpy
        img_0_np = img_0[:num_samples].detach().cpu().numpy()
        img_1_np = img_1[:num_samples].detach().cpu().numpy()
        img_recon_np = img_reconstructed[:num_samples].detach().cpu().numpy()

        # Create visualization
        fig, axes = plt.subplots(num_samples, 3, figsize=(10, 2 * num_samples))
        if num_samples == 1:
            axes = axes.reshape(1, -1)

        for i in range(num_samples):
            # Original
            axes[i, 0].imshow(img_0_np[i], cmap="gray")
            axes[i, 0].set_title("Original x(0)")
            axes[i, 0].axis("off")

            # Rotated
            axes[i, 1].imshow(img_1_np[i], cmap="gray")
            axes[i, 1].set_title("Rotated x(1)")
            axes[i, 1].axis("off")

            # Reconstructed
            axes[i, 2].imshow(img_recon_np[i], cmap="gray")
            axes[i, 2].set_title("Reconstructed x̂(0)")
            axes[i, 2].axis("off")

        plt.tight_layout()

        if save_path:
            os.makedirs(save_path, exist_ok=True)
            save_path_img = save_path + "/generated_panel.png"
            plt.savefig(save_path_img, dpi=150, bbox_inches="tight")
            print(f"Results saved to: {save_path_img}")

        plt.show()

    def get_validation_subset(
        self, num_samples: int = 800
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate test images with rotations."""
        if self.datamodule is None:
            self.setup_data()

        # Organize data by class
        class_ids = {i: [] for i in range(10)}
        for data in self.datamodule.val_dataloader().dataset:
            class_ids[data[1]].append(data[0])

        # Create validation subset
        mnist_subset = {c: torch.cat(v[:num_samples]) for c, v in class_ids.items()}
        return mnist_subset

    def process_batch(self, img_0, verbose=True):
        num_samples = len(img_0)
        # Generate random rotation angles
        angle_parameter = torch.randn_like(torch.empty(num_samples, dtype=torch.float))
        lambda_T = angle_parameter * (180 / 2)
        lambda_T = torch.where(
            lambda_T.abs() < 180 / 8, torch.ones_like(lambda_T) * 180 / 8, lambda_T
        )

        # Apply rotations
        img_1 = torch.stack(
            [
                ToTensor()(rotate_image(ToPILImage()(image), angle))
                for image, angle in zip(img_0, lambda_T)
            ]
        ).squeeze()

        output = self.sample_bridge(img_1, img_0, verbose)
        output["variances"] = lambda_T
        return output

    def run_test(self, save_path: Optional[str] = None) -> Dict[str, torch.Tensor]:
        """Run complete test pipeline."""
        print("Setting up MNIST Bridge Tester...")
        print(f"Device: {self.device}")
        print(f"Model path: {self.config.model_path}")
        print(f"T steps: {self.config.T}")

        # Load model and setup data
        self.load_model()
        self.setup_data()

        # Generate test images
        print("Generating test images...")
        img_0, img_1, angles = self.generate_test_images()
        print(f"Generated {len(img_0)} image pairs")

        # Sample from bridge
        print("Sampling from bridge model...")
        results = self.sample_bridge(img_1, img_0)

        # Visualize results
        print("Visualizing results...")
        self.visualize_results(results, save_path)

        return results


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Test MNIST Bridge Diffusion Model")
    parser.add_argument(
        "--model-path",
        type=str,
        default="experiments/BBDM/models/model_id_0_sin_epoch_99_T=1000.ckpt",
        help="Path to trained model",
    )
    parser.add_argument(
        "--use-conv", action="store_true", help="Use convolutional model"
    )
    parser.add_argument("--device", type=str, default="cpu", help="Device to use")
    parser.add_argument("--T", type=int, default=1000, help="Number of diffusion steps")
    parser.add_argument("--seed", type=int, default=5, help="Random seed")
    parser.add_argument("--save-path", type=str, help="Path to save results")
    parser.add_argument("--no-context", action="store_true", help="Disable context")
    parser.add_argument("--run-test", action="store_true", help="Run test")
    parser.add_argument(
        "--batch-size", type=int, default=64, help="Batch size for processing"
    )
    parser.add_argument("--stop-t", type=float, default=1.0, help="Stopping time")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Create configuration
    config = Config(
        model_path=args.model_path,
        use_conv=args.use_conv,
        device=args.device,
        context=not args.no_context,
        T=args.T,
        seed=args.seed,
        stop_t=args.stop_t,
    )

    # Run test
    tester = MNISTTester(config)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path, exist_ok=True)
        print(f"Created save directory: {args.save_path}")

    if args.run_test:
        print("Running MNIST Bridge Test...")
        results = tester.run_test(save_path=args.save_path)
        print("Testing completed successfully!")
    else:
        print("Running generation over all 10 MNIST digit classes from validation set")
        validation_subset = tester.get_validation_subset()
        tester.load_model()
        generated_dataset = {k: [] for k in validation_subset.keys()}
        for label, data in validation_subset.items():
            print(f"Label: {label}, Data: {data.shape}")
            batches = torch.split(data, args.batch_size)
            generated_results = []
            for i, batch in tqdm(
                enumerate(batches), total=len(batches), desc="Processing batches"
            ):
                res: Dict[str, torch.Tensor] = tester.process_batch(batch, verbose=True)
                generated_results.append(res)

            # Stack the results based on the key
            generated_results = {
                k: torch.concat([r[k] for r in generated_results])
                for k in generated_results[0].keys()
            }
            generated_dataset[label] = generated_results
        print("Generation completed successfully!")
        if not os.path.exists(args.save_path):
            os.makedirs(args.save_path, exist_ok=True)
        save_path = args.save_path + "/generated_valset.pth"
        torch.save(generated_dataset, save_path)
        print(f"Generated dataset saved to {save_path}")
