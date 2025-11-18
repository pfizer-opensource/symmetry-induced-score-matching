import torch
import sys
import numpy as np
import argparse
import os

# Add paths for imports
sys.path.append("..")
sys.path.append("../..")

from sism.mnist.train_classifier import MNISTClassifier


def create_batches(lst, batch_size):
    """Create batches from a list or tensor."""
    return [lst[i : i + batch_size] for i in range(0, len(lst), batch_size)]


def compute_frechet_inception_distance(features1, features2):
    """
    Compute Frechet Inception Distance between two sets of features.

    Args:
        features1 (torch.Tensor): Features from first distribution
        features2 (torch.Tensor): Features from second distribution

    Returns:
        torch.Tensor: FID score
    """
    if features1.ndim != 2:
        features1 = features1.view(features1.shape[0], -1)
    if features2.ndim != 2:
        features2 = features2.view(features2.shape[0], -1)

    mu1, sigma1 = torch.mean(features1, dim=0), torch.cov(features1.T)
    mu2, sigma2 = torch.mean(features2, dim=0), torch.cov(features2.T)

    mean_diff = mu1 - mu2
    mean_diff_squared = mean_diff.square().sum(dim=-1)

    # Calculate the sum of the traces of both covariance matrices
    trace_sum = sigma1.trace() + sigma2.trace()

    # Compute the eigenvalues of the matrix product of the covariance matrices
    sigma_mm = torch.matmul(sigma1, sigma2)
    eigenvals = torch.linalg.eigvals(sigma_mm)

    # Take the square root of each eigenvalue and take its sum
    sqrt_eigenvals_sum = eigenvals.sqrt().real.sum(dim=-1)

    # Calculate the FID
    fid = mean_diff_squared + trace_sum - 2 * sqrt_eigenvals_sum
    return fid


def evaluate_samples_from_model(
    sample_path: str,
    classifier,
    device,
    batch_size: int = 128,
    compute_fid: bool = True,
):
    """
    Evaluate samples from a model using the MNIST classifier and optionally compute FID.

    Args:
        sample_path (str): Path to the file containing samples.
        classifier: Trained classifier model
        device: PyTorch device
        batch_size (int): Batch size for processing
        compute_fid (bool): Whether to compute FID scores

    Returns:
        dict: Dictionary containing accuracy and optionally FID metrics
    """
    # Load samples
    print(f"Loading samples from: {sample_path}")
    samples = torch.load(sample_path, map_location="cpu")

    results = {}

    # Process each digit class
    for label in range(10):
        if label not in samples:
            print(f"Label {label} not found in samples, skipping...")
            continue

        img_true = samples[label]["img_0"]
        img_gen = samples[label]["img"]

        print(
            f"Processing label {label}: {len(img_true)} true, {len(img_gen)} generated images"
        )

        img_true_batches = create_batches(img_true, batch_size)
        img_gen_batches = create_batches(img_gen, batch_size)

        embeddings_true, embeddings_gen = [], []
        accuracies_true, accuracies_gen = [], []

        for true_subset, gen_subset in zip(img_true_batches, img_gen_batches):
            true_subset = true_subset.to(device)
            gen_subset = gen_subset.to(device)

            with torch.no_grad():
                if true_subset.dim() == 3:
                    true_subset = true_subset.unsqueeze(1)
                if gen_subset.dim() == 3:
                    gen_subset = gen_subset.unsqueeze(1)

                out_true = classifier(true_subset)
                out_gen = classifier(gen_subset)

                acc_true = (out_true["prediction"].argmax(-1) == label).float()
                acc_gen = (out_gen["prediction"].argmax(-1) == label).float()

                accuracies_true.append(acc_true)
                accuracies_gen.append(acc_gen)

                if compute_fid:
                    embeddings_true.append(out_true["embedding"])
                    embeddings_gen.append(out_gen["embedding"])

        accuracies_true = torch.cat(accuracies_true, dim=0)
        accuracies_gen = torch.cat(accuracies_gen, dim=0)

        if compute_fid:
            embeddings_true = torch.cat(embeddings_true, dim=0)
            embeddings_gen = torch.cat(embeddings_gen, dim=0)
            fid_score = compute_frechet_inception_distance(
                embeddings_true, embeddings_gen
            ).item()
        else:
            fid_score = np.nan

        results[label] = {
            "accuracy_true": accuracies_true.mean().item(),
            "accuracy_generated": accuracies_gen.mean().item(),
            "fid_score": fid_score,
        }

    # Print results
    print("\nResults:")
    print("-" * 80)
    for label, val in results.items():
        print(
            f"Label {label}: Accuracy True: {val['accuracy_true']:.4f}, "
            f"Accuracy Generated: {val['accuracy_generated']:.4f}, "
            f"FID Score: {val['fid_score']:.4f}"
        )

    return results


def load_classifier(classifier_path, device):
    """Load the trained MNIST classifier."""
    print(f"Loading classifier from: {classifier_path}")
    classifier = MNISTClassifier(num_classes=10)
    ckpt = torch.load(classifier_path, map_location="cpu")
    classifier.load_state_dict(ckpt["model_state_dict"])
    classifier = classifier.to(device)
    classifier.eval()
    return classifier


def compute_summary_statistics(results, method_name):
    """Compute and print summary statistics for the results."""
    accuracy_values = [v["accuracy_generated"] for v in results.values()]
    fid_values = [
        v["fid_score"] for v in results.values() if not np.isnan(v["fid_score"])
    ]

    accuracy_mean = np.mean(accuracy_values)
    accuracy_var = np.var(accuracy_values)

    print(f"\n{method_name} Summary:")
    print(f"Average Accuracy: {accuracy_mean:.4f} ± {np.sqrt(accuracy_var):.4f}")

    if fid_values:
        fid_mean = np.mean(fid_values)
        fid_var = np.var(fid_values)
        print(f"Average FID: {fid_mean:.4f} ± {np.sqrt(fid_var):.4f}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate generated MNIST samples")
    parser.add_argument(
        "--gsm-samples",
        type=str,
        default="experiments/GSM/generated_valset.pth",
        help="Path to GSM generated samples",
    )
    parser.add_argument(
        "--bridge-samples",
        type=str,
        default="experiments/BBDM/generated_valset.pth",
        help="Path to Bridge generated samples",
    )
    parser.add_argument(
        "--classifier",
        type=str,
        default="experiments/classifier/mnist_classifier_best.pth",
        help="Path to trained classifier",
    )
    parser.add_argument(
        "--batch-size", type=int, default=128, help="Batch size for evaluation"
    )
    parser.add_argument("--no-fid", action="store_true", help="Skip FID computation")
    parser.add_argument(
        "--device", type=str, default=None, help="Device to use (cuda/cpu)"
    )

    args = parser.parse_args()

    # Set device
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"Using device: {device}")

    # Load classifier
    classifier = load_classifier(args.classifier, device)

    # Evaluate GSM samples
    if os.path.exists(args.gsm_samples):
        print("\n" + "=" * 80)
        print("Evaluating GSM samples...")
        print("=" * 80)
        results_gsm = evaluate_samples_from_model(
            args.gsm_samples,
            classifier,
            device,
            batch_size=args.batch_size,
            compute_fid=not args.no_fid,
        )
        compute_summary_statistics(results_gsm, "GSM")
    else:
        print(f"GSM samples not found at: {args.gsm_samples}")

    # Evaluate Bridge samples
    if os.path.exists(args.bridge_samples):
        print("\n" + "=" * 80)
        print("Evaluating Bridge samples...")
        print("=" * 80)
        results_bridge = evaluate_samples_from_model(
            args.bridge_samples,
            classifier,
            device,
            batch_size=args.batch_size,
            compute_fid=not args.no_fid,
        )
        compute_summary_statistics(results_bridge, "Bridge")
    else:
        print(f"Bridge samples not found at: {args.bridge_samples}")


if __name__ == "__main__":
    main()
