from scipy.stats import wasserstein_distance_nd as wdist
import torch
import os
import json
import numpy as np
from torch import nn
import matplotlib.pyplot as plt


def create_score_network(
    input_dim: int, hidden_dim: int, output_dim: int, device: torch.device
) -> nn.Module:
    """Create a simple MLP score network."""
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.SiLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.SiLU(),
        nn.Linear(hidden_dim, output_dim),
    ).to(device)


def get_diffusion_coefficients(T, kind="linear-time", alpha_clip_min=0.001):
    if kind == "linear-time":
        t = np.linspace(1e-3, 1.0, T + 2)
        alphas_cumprod = 1.0 - t
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        alphas = alphas_cumprod[1:] / alphas_cumprod[:-1]
        betas = 1 - alphas
        betas = np.clip(betas, 0.0, 1.0 - alpha_clip_min)
        betas = torch.from_numpy(betas).float().squeeze()
    elif kind == "cosine":
        s = 0.008
        steps = T + 2
        x = torch.linspace(0, T, steps)
        alphas_cumprod = torch.cos(((x / T) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        alphas = alphas_cumprod[1:] / alphas_cumprod[:-1]
        alphas = alphas.clip(min=alpha_clip_min)
        betas = 1 - alphas
        betas = torch.clip(betas, 0.0, 1.0 - alpha_clip_min).float()
    elif kind == "ddpm":
        betas = torch.linspace(0.1 / T, 20 / T, T + 1)
    return betas


def save_losses(losses: list, method: str, dataset: str, save_dir: str) -> None:
    """Save training losses to JSON and plot."""
    # Create dataset-specific directory
    dataset_dir = os.path.join(save_dir, dataset)
    os.makedirs(dataset_dir, exist_ok=True)

    # Save losses as JSON
    losses_data = {
        "method": method,
        "dataset": dataset,
        "losses": losses,
        "final_loss": losses[-1] if losses else None,
        "min_loss": min(losses) if losses else None,
        "epochs": len(losses),
    }

    json_filename = os.path.join(dataset_dir, f"{method}_losses.json")
    with open(json_filename, "w") as f:
        json.dump(losses_data, f, indent=2)

    # Save loss plot
    plt.figure(figsize=(8, 6))
    plt.plot(losses, label=f"{method} Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{method} Training Loss - {dataset.capitalize()} Dataset")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plot_filename = os.path.join(dataset_dir, f"{method}_loss_curve.png")
    plt.savefig(plot_filename, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Losses saved to {json_filename}")
    print(f"Loss curve saved to {plot_filename}")


def gaussian_kernel(x: torch.Tensor, y: torch.Tensor, sigma: float) -> torch.Tensor:
    """
    Compute Gaussian (RBF) kernel between two sets of points.

    Args:
        x: Tensor of shape (n, d)
        y: Tensor of shape (m, d)
        sigma: Bandwidth parameter for the Gaussian kernel

    Returns:
        Kernel matrix of shape (n, m)
    """
    # Ensure sigma is positive to avoid division by zero
    sigma = max(sigma, 1e-8)

    # Compute pairwise squared distances with numerical stability
    x_norm = (x**2).sum(dim=1, keepdim=True)  # (n, 1)
    y_norm = (y**2).sum(dim=1, keepdim=True)  # (m, 1)

    # ||x - y||^2 = ||x||^2 + ||y||^2 - 2 * x^T * y
    dist_sq = x_norm + y_norm.t() - 2 * torch.mm(x, y.t())

    # Ensure non-negative distances (numerical stability)
    dist_sq = torch.clamp(dist_sq, min=0.0)

    # Compute kernel with numerical stability
    # Avoid overflow by clamping the exponent argument
    exponent = -dist_sq / (2 * sigma**2)
    exponent = torch.clamp(exponent, min=-50, max=50)  # Prevent overflow/underflow

    kernel = torch.exp(exponent)

    # Check for NaN or inf in result
    if torch.isnan(kernel).any() or torch.isinf(kernel).any():
        print("Warning: NaN or inf detected in kernel computation")
        # Return identity-like kernel as fallback
        n, m = x.shape[0], y.shape[0]
        if n == m and torch.allclose(x, y):
            return torch.eye(n, device=x.device, dtype=x.dtype)
        else:
            return torch.ones(n, m, device=x.device, dtype=x.dtype) * 1e-8

    return kernel


def compute_mmd(x: torch.Tensor, y: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    """
    Compute Maximum Mean Discrepancy (MMD) between two distributions.

    Args:
        x: Samples from first distribution, shape (n, d)
        y: Samples from second distribution, shape (m, d)
        sigma: Bandwidth parameter for Gaussian kernel

    Returns:
        MMD distance (scalar tensor)
    """
    # Remove NaN values
    x_clean = x[~torch.isnan(x).any(dim=1)]
    y_clean = y[~torch.isnan(y).any(dim=1)]

    # Check if we have enough samples after cleaning
    if len(x_clean) < 2 or len(y_clean) < 2:
        print(
            f"Warning: Insufficient clean samples after NaN removal. x: {len(x_clean)}, y: {len(y_clean)}"
        )
        return torch.tensor(float("inf"))

    # Remove infinite values
    x_clean = x_clean[~torch.isinf(x_clean).any(dim=1)]
    y_clean = y_clean[~torch.isinf(y_clean).any(dim=1)]

    if len(x_clean) < 2 or len(y_clean) < 2:
        print(
            f"Warning: Insufficient clean samples after inf removal. x: {len(x_clean)}, y: {len(y_clean)}"
        )
        return torch.tensor(float("inf"))

    # Ensure tensors are on the same device and dtype
    if x_clean.device != y_clean.device:
        y_clean = y_clean.to(x_clean.device)
    if x_clean.dtype != y_clean.dtype:
        y_clean = y_clean.to(x_clean.dtype)

    n, m = x_clean.shape[0], y_clean.shape[0]

    # Compute kernel matrices with numerical stability
    try:
        k_xx = gaussian_kernel(x_clean, x_clean, sigma)
        k_yy = gaussian_kernel(y_clean, y_clean, sigma)
        k_xy = gaussian_kernel(x_clean, y_clean, sigma)

        # Check for NaN in kernel matrices
        if (
            torch.isnan(k_xx).any()
            or torch.isnan(k_yy).any()
            or torch.isnan(k_xy).any()
        ):
            print("Warning: NaN detected in kernel matrices")
            return torch.tensor(float("inf"))

        # MMD^2 = E[k(X,X)] + E[k(Y,Y)] - 2*E[k(X,Y)]
        # Use numerically stable computation
        k_xx_sum = k_xx.sum() - torch.diag(k_xx).sum()
        k_yy_sum = k_yy.sum() - torch.diag(k_yy).sum()
        k_xy_sum = k_xy.sum()

        mmd_sq = (
            k_xx_sum / (n * (n - 1)) + k_yy_sum / (m * (m - 1)) - 2 * k_xy_sum / (n * m)
        )

        # Ensure non-negative before taking square root
        mmd_sq = torch.clamp(mmd_sq, min=0.0)

        # Check for NaN in final result
        if torch.isnan(mmd_sq):
            print("Warning: NaN in MMD squared computation")
            return torch.tensor(float("inf"))

        return torch.sqrt(mmd_sq)

    except Exception as e:
        print(f"Error in MMD computation: {e}")
        return torch.tensor(float("inf"))


def compute_wdist(original_data: torch.Tensor, sampled_data: torch.Tensor) -> float:
    """
    Compute Wasserstein distance between two distributions.

    Args:
        original_data: Ground truth data, shape (n, d)
        sampled_data: Generated samples, shape (m, d)

    Returns:
        Wasserstein distance (scalar)
    """
    # Remove NaN values
    original_clean = original_data[~torch.isnan(original_data).any(dim=1)]
    sampled_clean = sampled_data[~torch.isnan(sampled_data).any(dim=1)]

    if len(original_clean) < 2 or len(sampled_clean) < 2:
        print(
            f"Warning: Insufficient clean samples for Wasserstein distance. "
            f"Original: {len(original_clean)}, Sampled: {len(sampled_clean)}"
        )
        return float("inf")

    # Convert to numpy for scipy computation
    original_np = original_clean.numpy()
    sampled_np = sampled_clean.numpy()

    # Compute Wasserstein distance
    try:
        return wdist(original_np, sampled_np)
    except Exception as e:
        print(f"Error computing Wasserstein distance: {e}")
        return float("inf")


def compute_mmd_multi_scale(
    x: torch.Tensor, y: torch.Tensor, sigmas: list | None = None
) -> torch.Tensor:
    """
    Compute MMD with multiple bandwidth parameters and take the average.
    This provides a more robust estimate by considering different scales.

    Args:
        x: Samples from first distribution, shape (n, d)
        y: Samples from second distribution, shape (m, d)
        sigmas: List of bandwidth parameters. If None, uses heuristic values.

    Returns:
        Multi-scale MMD distance (scalar tensor)
    """
    # Remove NaN values first
    x_clean = x[~torch.isnan(x).any(dim=1)]
    y_clean = y[~torch.isnan(y).any(dim=1)]

    if len(x_clean) < 2 or len(y_clean) < 2:
        print(
            f"Warning: Insufficient clean samples for multi-scale MMD. x: {len(x_clean)}, y: {len(y_clean)}"
        )
        return torch.tensor(float("inf"))

    if sigmas is None:
        try:
            # Heuristic: use median pairwise distance and its multiples
            combined = torch.cat([x_clean, y_clean], dim=0)
            pdist = torch.pdist(combined)

            # Remove NaN distances
            pdist_clean = pdist[~torch.isnan(pdist)]
            if len(pdist_clean) == 0:
                print("Warning: All pairwise distances are NaN")
                return torch.tensor(float("inf"))

            median_dist = torch.median(pdist_clean)

            # Ensure median distance is positive
            if median_dist <= 0:
                median_dist = torch.tensor(1.0)

            sigmas = [median_dist * scale for scale in [0.1, 0.5, 1.0, 2.0, 5.0]]
        except Exception as e:
            print(f"Error computing heuristic sigmas: {e}")
            # Fallback to fixed sigmas
            sigmas = [0.1, 0.5, 1.0, 2.0, 5.0]

    mmd_values = []
    for sigma in sigmas:
        try:
            mmd_val = compute_mmd(x_clean, y_clean, float(sigma))
            # Only include finite values
            if torch.isfinite(mmd_val):
                mmd_values.append(mmd_val)
        except Exception as e:
            print(f"Error computing MMD with sigma={sigma}: {e}")
            continue

    if len(mmd_values) == 0:
        print("Warning: No valid MMD values computed")
        return torch.tensor(float("inf"))

    return torch.stack(mmd_values).mean()


def evaluate_sample_quality(
    original_data: torch.Tensor,
    sampled_data: torch.Tensor,
    method: str,
    dataset: str,
    save_dir: str,
) -> dict:
    """
    Evaluate the quality of generated samples using multiple metrics.

    Args:
        original_data: Ground truth data, shape (n, d)
        sampled_data: Generated samples, shape (m, d)
        method: Name of the generation method (e.g., "GSM", "Fisher")
        dataset: Name of the dataset
        save_dir: Directory to save results

    Returns:
        Dictionary containing evaluation metrics
    """
    # Create dataset-specific directory
    dataset_dir = os.path.join(save_dir, dataset)
    os.makedirs(dataset_dir, exist_ok=True)

    # Convert to tensors if needed
    if isinstance(original_data, np.ndarray):
        original_data = torch.from_numpy(original_data).float()
    if isinstance(sampled_data, np.ndarray):
        sampled_data = torch.from_numpy(sampled_data).float()

    # Check for and report NaN/inf values
    orig_nan_count = torch.isnan(original_data).sum().item()
    samp_nan_count = torch.isnan(sampled_data).sum().item()
    orig_inf_count = torch.isinf(original_data).sum().item()
    samp_inf_count = torch.isinf(sampled_data).sum().item()

    if orig_nan_count > 0 or samp_nan_count > 0:
        print(
            f"Warning: NaN values detected - Original: {orig_nan_count}, Sampled: {samp_nan_count}"
        )
    if orig_inf_count > 0 or samp_inf_count > 0:
        print(
            f"Warning: Inf values detected - Original: {orig_inf_count}, Sampled: {samp_inf_count}"
        )

    # Compute metrics with error handling
    try:
        mmd_single = compute_mmd(original_data, sampled_data, sigma=1.0)
        if not torch.isfinite(mmd_single):
            print(f"Warning: MMD single-scale returned non-finite value: {mmd_single}")
            mmd_single = torch.tensor(float("inf"))
    except Exception as e:
        print(f"Error computing single-scale MMD: {e}")
        mmd_single = torch.tensor(float("inf"))

    try:
        mmd_multi = compute_mmd_multi_scale(original_data, sampled_data)
        if not torch.isfinite(mmd_multi):
            print(f"Warning: MMD multi-scale returned non-finite value: {mmd_multi}")
            mmd_multi = torch.tensor(float("inf"))
    except Exception as e:
        print(f"Error computing multi-scale MMD: {e}")
        mmd_multi = torch.tensor(float("inf"))

    # Compute basic statistics for comparison with NaN handling
    try:
        orig_mean = torch.nanmean(original_data, dim=0)
        samp_mean = torch.nanmean(sampled_data, dim=0)
        mean_diff = torch.norm(orig_mean - samp_mean)

        if not torch.isfinite(mean_diff):
            print(f"Warning: Mean difference is non-finite: {mean_diff}")
            mean_diff = torch.tensor(float("inf"))
    except Exception as e:
        print(f"Error computing mean difference: {e}")
        mean_diff = torch.tensor(float("inf"))

    try:
        # Use nanstd equivalent (manual computation)
        orig_std = torch.sqrt(
            torch.nanmean((original_data - orig_mean.unsqueeze(0)) ** 2, dim=0)
        )
        samp_std = torch.sqrt(
            torch.nanmean((sampled_data - samp_mean.unsqueeze(0)) ** 2, dim=0)
        )
        std_diff = torch.norm(orig_std - samp_std)

        if not torch.isfinite(std_diff):
            print(f"Warning: Std difference is non-finite: {std_diff}")
            std_diff = torch.tensor(float("inf"))
    except Exception as e:
        print(f"Error computing std difference: {e}")
        std_diff = torch.tensor(float("inf"))

    try:
        original_data_subsampled = original_data[
            torch.randperm(original_data.size(0))[:1000]
        ]
        sampled_data_subsampled = sampled_data[
            torch.randperm(sampled_data.size(0))[:1000]
        ]
        # Compute Wasserstein distance
        wdist_value = compute_wdist(original_data_subsampled, sampled_data_subsampled)
        if not np.isfinite(wdist_value):
            print(
                f"Warning: Wasserstein distance returned non-finite value: {wdist_value}"
            )
            wdist_value = float("inf")
    except Exception as e:
        print(f"Error computing Wasserstein distance: {e}")
        wdist_value = float("inf")

    metrics = {
        "method": method,
        "dataset": dataset,
        "mmd_single_scale": (
            float(mmd_single.item()) if torch.isfinite(mmd_single) else float("inf")
        ),
        "mmd_multi_scale": (
            float(mmd_multi.item()) if torch.isfinite(mmd_multi) else float("inf")
        ),
        "mean_difference": (
            float(mean_diff.item()) if torch.isfinite(mean_diff) else float("inf")
        ),
        "std_difference": (
            float(std_diff.item()) if torch.isfinite(std_diff) else float("inf")
        ),
        "n_original": original_data.shape[0],
        "n_sampled": sampled_data.shape[0],
        "original_nan_count": orig_nan_count,
        "sampled_nan_count": samp_nan_count,
        "original_inf_count": orig_inf_count,
        "sampled_inf_count": samp_inf_count,
        "wasserstein_distance": wdist_value,
    }

    # Save metrics to JSON
    metrics_filename = os.path.join(dataset_dir, f"{method}_quality_metrics.json")
    with open(metrics_filename, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Quality metrics for {method} on {dataset}:")
    print(f"  MMD (single-scale): {mmd_single.item():.6f}")
    print(f"  MMD (multi-scale): {mmd_multi.item():.6f}")
    print(f"  Mean difference: {mean_diff.item():.6f}")
    print(f"  Std difference: {std_diff.item():.6f}")
    print(f"  Wasserstein distance: {wdist_value:.6f}")
    if (
        orig_nan_count > 0
        or samp_nan_count > 0
        or orig_inf_count > 0
        or samp_inf_count > 0
    ):
        print(
            f"  Data quality: NaN (orig: {orig_nan_count}, samp: {samp_nan_count}), Inf (orig: {orig_inf_count}, samp: {samp_inf_count})"
        )
    print(f"  Metrics saved to: {metrics_filename}")

    return metrics


def generate_comparison_report(
    gsm_metrics: dict, fisher_metrics: dict, dataset: str, save_dir: str
) -> None:
    """
    Generate a comparison report between GSM and Fisher methods.

    Args:
        gsm_metrics: Metrics dictionary for GSM method
        fisher_metrics: Metrics dictionary for Fisher method
        dataset: Dataset name
        save_dir: Directory to save the report
    """
    # Create dataset-specific directory
    dataset_dir = os.path.join(save_dir, dataset)
    os.makedirs(dataset_dir, exist_ok=True)

    # Calculate relative differences and improvements
    def calculate_relative_diff(gsm_val, fisher_val):
        """Calculate relative difference between two values."""
        if fisher_val == 0:
            return float("inf") if gsm_val != 0 else 0.0
        return (gsm_val - fisher_val) / abs(fisher_val) * 100

    mmd_single_diff = calculate_relative_diff(
        gsm_metrics["mmd_single_scale"], fisher_metrics["mmd_single_scale"]
    )
    mmd_multi_diff = calculate_relative_diff(
        gsm_metrics["mmd_multi_scale"], fisher_metrics["mmd_multi_scale"]
    )
    mean_diff = calculate_relative_diff(
        gsm_metrics["mean_difference"], fisher_metrics["mean_difference"]
    )
    std_diff = calculate_relative_diff(
        gsm_metrics["std_difference"], fisher_metrics["std_difference"]
    )
    wdist_diff = calculate_relative_diff(
        gsm_metrics["wasserstein_distance"], fisher_metrics["wasserstein_distance"]
    )

    report = {
        "dataset": dataset,
        "comparison": {
            "mmd_single_scale": {
                "GSM": gsm_metrics["mmd_single_scale"],
                "Fisher": fisher_metrics["mmd_single_scale"],
                "relative_difference_percent": mmd_single_diff,
            },
            "mmd_multi_scale": {
                "GSM": gsm_metrics["mmd_multi_scale"],
                "Fisher": fisher_metrics["mmd_multi_scale"],
                "relative_difference_percent": mmd_multi_diff,
            },
            "mean_difference": {
                "GSM": gsm_metrics["mean_difference"],
                "Fisher": fisher_metrics["mean_difference"],
                "relative_difference_percent": mean_diff,
            },
            "std_difference": {
                "GSM": gsm_metrics["std_difference"],
                "Fisher": fisher_metrics["std_difference"],
                "relative_difference_percent": std_diff,
            },
            "wasserstein_distance": {
                "GSM": gsm_metrics["wasserstein_distance"],
                "Fisher": fisher_metrics["wasserstein_distance"],
                "relative_difference_percent": wdist_diff,
            },
        },
        "summary": {
            "gsm_better_metrics": sum(
                1
                for diff in [
                    mmd_single_diff,
                    mmd_multi_diff,
                    mean_diff,
                    std_diff,
                    wdist_diff,
                ]
                if diff < 0
            ),
            "fisher_better_metrics": sum(
                1
                for diff in [
                    mmd_single_diff,
                    mmd_multi_diff,
                    mean_diff,
                    std_diff,
                    wdist_diff,
                ]
                if diff > 0
            ),
            "equivalent_metrics": sum(
                1
                for diff in [
                    mmd_single_diff,
                    mmd_multi_diff,
                    mean_diff,
                    std_diff,
                    wdist_diff,
                ]
                if diff == 0
            ),
        },
    }

    # Save report
    report_filename = os.path.join(dataset_dir, "method_comparison.json")
    with open(report_filename, "w") as f:
        json.dump(report, f, indent=2)

    # Print summary
    print(f"\n=== Method Comparison for {dataset.capitalize()} Dataset ===")
    print(
        f"MMD (single-scale): GSM={gsm_metrics['mmd_single_scale']:.6f}, Fisher={fisher_metrics['mmd_single_scale']:.6f} (GSM {mmd_single_diff:+.1f}%)"
    )
    print(
        f"MMD (multi-scale):  GSM={gsm_metrics['mmd_multi_scale']:.6f}, Fisher={fisher_metrics['mmd_multi_scale']:.6f} (GSM {mmd_multi_diff:+.1f}%)"
    )
    print(
        f"Mean difference:    GSM={gsm_metrics['mean_difference']:.6f}, Fisher={fisher_metrics['mean_difference']:.6f} (GSM {mean_diff:+.1f}%)"
    )
    print(
        f"Std difference:     GSM={gsm_metrics['std_difference']:.6f}, Fisher={fisher_metrics['std_difference']:.6f} (GSM {std_diff:+.1f}%)"
    )
    print(
        f"Wasserstein dist:   GSM={gsm_metrics['wasserstein_distance']:.6f}, Fisher={fisher_metrics['wasserstein_distance']:.6f} (GSM {wdist_diff:+.1f}%)"
    )
    print(
        f"Summary: GSM performs better on {report['summary']['gsm_better_metrics']} metrics, "
        f"Fisher performs better on {report['summary']['fisher_better_metrics']} metrics"
    )
    print(f"Comparison report saved to: {report_filename}")


def create_overall_summary(datasets: list, save_dir: str) -> None:
    """
    Create an overall summary of all experiments across datasets.

    Args:
        datasets: List of dataset names that were processed
        save_dir: Main results directory
    """
    overall_summary = {
        "experiment_info": {
            "datasets_processed": datasets,
            "total_datasets": len(datasets),
            "experiment_timestamp": None,  # Could add timestamp if needed
        },
        "results_by_dataset": {},
        "overall_statistics": {
            "total_gsm_better": 0,
            "total_fisher_better": 0,
            "total_equivalent": 0,
            "datasets_processed": 0,
        },
    }

    # Collect results from each dataset
    for dataset in datasets:
        dataset_dir = os.path.join(save_dir, dataset)
        comparison_file = os.path.join(dataset_dir, "method_comparison.json")

        if os.path.exists(comparison_file):
            try:
                with open(comparison_file, "r") as f:
                    comparison_data = json.load(f)

                overall_summary["results_by_dataset"][dataset] = {
                    "gsm_better_metrics": comparison_data["summary"][
                        "gsm_better_metrics"
                    ],
                    "fisher_better_metrics": comparison_data["summary"][
                        "fisher_better_metrics"
                    ],
                    "equivalent_metrics": comparison_data["summary"][
                        "equivalent_metrics"
                    ],
                    "metrics": comparison_data["comparison"],
                }

                # Update overall statistics
                overall_summary["overall_statistics"][
                    "total_gsm_better"
                ] += comparison_data["summary"]["gsm_better_metrics"]
                overall_summary["overall_statistics"][
                    "total_fisher_better"
                ] += comparison_data["summary"]["fisher_better_metrics"]
                overall_summary["overall_statistics"][
                    "total_equivalent"
                ] += comparison_data["summary"]["equivalent_metrics"]
                overall_summary["overall_statistics"]["datasets_processed"] += 1

            except (json.JSONDecodeError, KeyError) as e:
                print(f"Warning: Could not read comparison data for {dataset}: {e}")

    # Save overall summary
    summary_filename = os.path.join(save_dir, "overall_experiment_summary.json")
    with open(summary_filename, "w") as f:
        json.dump(overall_summary, f, indent=2)

    # Print overall summary
    stats = overall_summary["overall_statistics"]
    total_metrics = (
        stats["total_gsm_better"]
        + stats["total_fisher_better"]
        + stats["total_equivalent"]
    )

    print(f"\n{'='*60}")
    print("OVERALL EXPERIMENT SUMMARY")
    print(f"{'='*60}")
    print(f"Datasets processed: {', '.join(datasets)}")
    print(f"Total datasets: {len(datasets)}")
    print(f"Total metrics compared: {total_metrics}")
    print("\nOverall Performance Comparison:")
    print(
        f"  GSM performs better: {stats['total_gsm_better']} metrics ({stats['total_gsm_better']/max(total_metrics, 1)*100:.1f}%)"
    )
    print(
        f"  Fisher performs better: {stats['total_fisher_better']} metrics ({stats['total_fisher_better']/max(total_metrics, 1)*100:.1f}%)"
    )
    print(
        f"  Equivalent performance: {stats['total_equivalent']} metrics ({stats['total_equivalent']/max(total_metrics, 1)*100:.1f}%)"
    )

    # Overall trend analysis
    if stats["total_gsm_better"] > stats["total_fisher_better"]:
        print("  Overall trend: GSM shows better performance across more metrics")
    elif stats["total_fisher_better"] > stats["total_gsm_better"]:
        print("  Overall trend: Fisher shows better performance across more metrics")
    else:
        print("  Overall trend: Both methods show equivalent performance")

    print(f"\nDetailed summary saved to: {summary_filename}")
    print(f"{'='*60}")
