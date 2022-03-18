import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, random_split, Subset
from torchvision import datasets, transforms


class ExplanationDataset(Dataset):
    """An Abstract class for datasets with explanations
    """
    def get_data(self, idx):
        raise NotImplementedError

    def get_explanations(self, idx):
        # If not overloaded, return nan
        return np.nan

    def get_spatial_explanations(self, idx):
        # If not overloaded, return nan
        return np.nan

    def __getitem__(self, idx):
        inputs, labels = self.get_data(idx)
        return (inputs,
                self.get_explanations(idx),
                self.get_spatial_explanations(idx),
                labels)


def split_dataset(dataset, n_samples):
    """Splits a dataset in two: a first dataset consisting of the first `n_samples`
        samples, and the second consisting of the remaining ones
    """
    assert n_samples <= len(dataset), "The length of dataset has to be shorter than n_samples"
    sub_idx_1 = list(range(n_samples))
    sub_idx_2 = list(range(n_samples, len(dataset)))
    return Subset(dataset, sub_idx_1), Subset(dataset, sub_idx_2)


class ExplainLabelBinaryDataset(ExplanationDataset):
    def __init__(self, dataset, num_classes=10):
        self.num_classes = num_classes
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def get_data(self, idx):
        inputs, digit = self.dataset[idx]
        label = digit % 2
        return inputs, label

    def get_explanations(self, idx):
        _, digit = self.dataset[idx]
        explanation_onehot = F.one_hot(torch.tensor(digit).long(), num_classes=self.num_classes).float()
        return explanation_onehot


def explanation_mnist_dataset(n_train_samples, data_dir='~/data/', conv_input=False, random_train_subset=False):
    """train dataset is `n_train_samples` samples at random from mnist training set,
       val dataset is the last 5k samples of the training set
       test dataset is the regulad mnist 10k samples dataset

        Args:
            conv_input (bool): Whether the inputs should maintain 2D structure or be permutation invariant
    """
    transform = [transforms.ToTensor(),
                 transforms.Normalize((0.1307,), (0.3081,))]
    if not conv_input:
        transform.append(transforms.Lambda(lambda x: x.view(x.size(1) * x.size(2))))
    transform = transforms.Compose(transform)

    full_train_ds, val_ds = split_dataset(datasets.MNIST(data_dir, train=True, download=True,
                                                         transform=transform), 55000)
    if random_train_subset:
        train_ds, _ = random_split(full_train_ds, [n_train_samples, len(full_train_ds) - n_train_samples])
    else:
        train_ds, _ = split_dataset(full_train_ds, n_train_samples)

    test_ds = datasets.MNIST(data_dir, train=False, transform=transform)
    return ExplainLabelBinaryDataset(train_ds), ExplainLabelBinaryDataset(val_ds), ExplainLabelBinaryDataset(test_ds)
