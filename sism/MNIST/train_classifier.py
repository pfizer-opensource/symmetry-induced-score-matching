# Importing dependencies
import torch
from torch import nn
from torch.optim import Adam
import sys

sys.path.append("..")
sys.path.append("../..")
try:
    from sism.data.mnist_data import MNISTDataModule
except:
    from mnist_data import MNISTDataModule

import os
import argparse
from tqdm import tqdm


class MNISTClassifier(nn.Module):
    """Simple CNN classifier for MNIST digits."""

    def __init__(self, num_classes=10):
        super(MNISTClassifier, self).__init__()

        # Convolutional layers
        self.conv_layers = nn.Sequential(
            # First conv block
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.1),
            # Second conv block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        embedding = self.conv_layers(x)
        prediction = self.fc_layers(embedding)
        out = {"embedding": embedding, "prediction": prediction}
        return out


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train the model for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(train_loader, desc="Training")

    for batch_idx, (data, target) in enumerate(progress_bar):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output["prediction"], target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        predicted = torch.argmax(output["prediction"], 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

        # Update progress bar
        progress_bar.set_postfix(
            {
                "Loss": f"{running_loss/(batch_idx+1):.4f}",
                "Acc": f"{100.*correct/total:.2f}%",
            }
        )

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100.0 * correct / total

    return epoch_loss, epoch_acc


def evaluate(model, test_loader, criterion, device):
    """Evaluate the model on test set."""
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in tqdm(test_loader, desc="Testing"):
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output["prediction"], target).item()

            predicted = torch.argmax(output["prediction"], 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    test_loss /= len(test_loader)
    test_acc = 100.0 * correct / total

    return test_loss, test_acc


def main():
    parser = argparse.ArgumentParser(description="Train MNIST Classifier")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=10,
        metavar="N",
        help="number of epochs to train (default: 10)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        metavar="LR",
        help="learning rate (default: 0.001)",
    )
    parser.add_argument(
        "--save_model", action="store_true", default=True, help="save the trained model"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./experiments/classifier",
        help="directory to save trained models",
    )

    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Get data loaders
    print("Loading MNIST data...")
    datamodule = MNISTDataModule(affine=False, batch_size=args.batch_size)
    datamodule.setup()
    train_loader = datamodule.train_dataloader()
    test_loader = datamodule.test_dataloader()

    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")

    # Initialize model
    model = MNISTClassifier(num_classes=10).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=args.lr)

    # Training loop
    print(f"\nStarting training for {args.num_epochs} epochs...")
    best_acc = 0.0

    for epoch in range(args.num_epochs):
        print(f"\nEpoch {epoch+1}/{args.num_epochs}")

        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # Evaluate
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")

        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            if args.save_model:
                model_path = os.path.join(args.save_dir, "mnist_classifier_best.pth")
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "test_acc": test_acc,
                        "test_loss": test_loss,
                    },
                    model_path,
                )
                print(f"Best model saved: {model_path}")

    print(f"\nTraining completed! Best test accuracy: {best_acc:.2f}%")

    # Save final model
    if args.save_model:
        final_model_path = os.path.join(args.save_dir, "mnist_classifier_final.pth")
        torch.save(
            {
                "epoch": args.num_epochs - 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "test_acc": test_acc,
                "test_loss": test_loss,
            },
            final_model_path,
        )
        print(f"Final model saved: {final_model_path}")


if __name__ == "__main__":
    main()
