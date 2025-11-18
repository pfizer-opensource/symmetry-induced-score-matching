from typing import Tuple, Any

import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pytorch_lightning import LightningDataModule
from torchvision.datasets import MNIST


class RotatedMNISTDataset(MNIST):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        super().__init__(root, train, transform, target_transform, download)

    def rotate_image(self, angle):

        rotate_transform = transforms.RandomAffine(degrees=(-angle, angle))

        return rotate_transform

    def __getitem__(self, index: int) -> Tuple[Any, Any, Any]:
        img, target = self.data[index], int(self.targets[index])
        img = Image.fromarray(img.numpy(), mode="L")

        angle = torch.rand(1).item() * 360  # random angle between 0 and 360 degrees

        rotate_transform = self.rotate_image(angle)
        img_rot = rotate_transform(img)

        if self.transform is not None:
            img = self.transform(img)
            img_rot = self.transform(img_rot)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, img_rot, target


class MNISTDataModule(LightningDataModule):
    def __init__(self, data_dir: str = './', batch_size: int = 64, affine: bool = False, num_workers=0):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transform = transforms.Compose([
             transforms.ToTensor(),
        #     transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.num_workers = num_workers

        if affine:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.RandomAffine(translate=(0.3, 0.3), degrees=(-180, 180), scale=(0.5, 1.4))
            ])

    def setup(self, stage=None):
        if stage in ('fit', None):
            self.mnist_train = datasets.MNIST(self.data_dir, train=True, download=True, transform=self.transform)
            self.mnist_val = datasets.MNIST(self.data_dir, train=False, download=True, transform=self.transform)
        if stage in ('test', None):
            self.mnist_test = datasets.MNIST(self.data_dir, train=False, download=True, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    mnist_with_affine = MNISTDataModule(affine=True)
    mnist_without_affine = MNISTDataModule(affine=False)

    mnist_with_affine.setup()
    mnist_without_affine.setup()

    dataloader_with_affine = mnist_with_affine.train_dataloader()
    dataloader_without_affine = mnist_without_affine.train_dataloader()

    examples_with_affine, _ = next(iter(dataloader_with_affine))
    examples_without_affine, _ = next(iter(dataloader_without_affine))

    fig, axs = plt.subplots(2, 5, figsize=(10, 4))  # Change the numbers here to plot more or fewer images
    for i in range(5):
        axs[0, i].imshow(examples_with_affine[i].squeeze(), cmap='gray')
        axs[0, i].set_title('With affine')
        axs[0, i].axis('off')

        axs[1, i].imshow(examples_without_affine[i].squeeze(), cmap='gray')
        axs[1, i].set_title('Without affine')
        axs[1, i].axis('off')

    plt.show()

    # Create an instance of RotatedMNISTDataset
    dataset = RotatedMNISTDataset(root='./', download=True)

    # Get a few items from the dataset
    for i in range(5):
        img, img_rot, target = dataset[i]  # 30 degrees rotation

        # Plot the original and rotated images
        fig, axs = plt.subplots(1, 3, figsize=(5, 2))
        axs[0].imshow(img, cmap='gray')
        axs[0].set_title(f'Original, Label: {target}')
        axs[0].axis('off')

        axs[1].imshow(img_rot, cmap='gray')
        axs[1].set_title(f'Rotated, Label: {target}')
        axs[1].axis('off')

        angle = torch.rand(1).item() * 360  # random angle between 0 and 360 degrees
        rotate_transform = transforms.RandomAffine(degrees=(-angle, angle))
        img_rot2 = rotate_transform(img)

        axs[2].imshow(img_rot2, cmap='gray')
        axs[2].set_title(f'Rotated, Label: {target}')
        axs[2].axis('off')

        plt.show()