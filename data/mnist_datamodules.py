import pytorch_lightning as pl
from torch.utils.data import DataLoader

from .mnist_dataset import explanation_mnist_dataset


class ExplanationMNIST(pl.LightningDataModule):
    def __init__(
        self,
        n_train_samples: int = 1000,
        conv_input: bool = True,
        batch_size: int = 32,
        num_workers: int = 8,
        **kwargs
    ):
        super().__init__()
        self.n_train_samples = n_train_samples
        self.num_classes = 2
        self.conv_input = conv_input
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        trainset, valset, testset = explanation_mnist_dataset(
            n_train_samples=self.n_train_samples, conv_input=self.conv_input
        )
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            self.mnist_train = trainset
            self.mnist_val = valset

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.mnist_test = testset

        # self.dims is returned when you call dm.size()
        self.dims = trainset[0][0].shape

    def train_dataloader(self):
        return DataLoader(
            self.mnist_train,
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.mnist_val, shuffle=False, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.mnist_test,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.mnist_test,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
