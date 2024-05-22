from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, Subset
import math


def label_map_one_minus_one(y):
    if y in {0, 1, 2, 3, 4}:
        return 1
    return -1


def label_map_zero_one(y):
    if y in {0, 1, 2, 3, 4}:
        return 0
    return 1


def load_data(label_map):
    image_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    """Load MNIST and transform to binary classification"""
    train_data = datasets.MNIST(
        "./data",
        train=True,
        transform=image_transform,
        # transform=transforms.ToTensor(),  # TODO: Note, this is what you had for the multiclass experiment (i.e. you forgot to normalize)
        download=True,
        target_transform=label_map,
    )
    test_data = datasets.MNIST(
        "./data",
        train=False,
        transform=image_transform,
        # transform=transforms.ToTensor(),
        download=True,
        target_transform=label_map,
    )
    return train_data, test_data


def split_data(dataset: Dataset, prior_prop: float, val_prop: float, cert_prop: float):
    assert prior_prop + val_prop + cert_prop == 1.0
    num_samples = len(dataset)
    n_prior = math.ceil(prior_prop * num_samples)
    n_val = math.ceil(val_prop * num_samples)

    prior_indices = list(range(n_prior))
    val_indices = list(range(n_prior, n_prior + n_val))
    cert_indices = list(range(n_prior + n_val, num_samples))

    prior_data = Subset(dataset, prior_indices)
    val_data = Subset(dataset, val_indices)
    cert_data = Subset(dataset, cert_indices)
    post_data = dataset

    return prior_data, val_data, cert_data, post_data


def take_subsets(datasets, subset_size):
    return [Subset(dataset, list(range(subset_size))) for dataset in datasets]


def make_dataloaders(datasets, batch_size, shuffle=True):
    return [
        DataLoader(dataset, batch_size=batch_size, shuffle=shuffle) for dataset in datasets
    ]
