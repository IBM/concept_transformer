import albumentations as A
import pytorch_lightning as pl
from albumentations.augmentations.geometric.resize import Resize
from albumentations.augmentations.geometric.rotate import Rotate
from albumentations.augmentations.transforms import HorizontalFlip, Normalize
from albumentations.pytorch.transforms import ToTensorV2
from torch.utils.data import DataLoader, random_split

from .cub2011parts import CUB2011Parts_dataset


class CUB2011Parts(pl.LightningDataModule):

    def __init__(self, batch_size: int = 32, num_workers: int = 8,
                 data_dir: str = "~/data/cub2011", **kwargs):
        super().__init__()

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_transform = A.Compose([Resize(224, 224),
                                          HorizontalFlip(p=0.5),
                                          Rotate(limit=(-30,30),p=1.0),
                                          Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                                          ToTensorV2()],
                                         keypoint_params = A.KeypointParams(format='xy', remove_invisible=False))

        self.test_transform = A.Compose([Resize(224, 224),
                                         Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                                         ToTensorV2()],
                                        keypoint_params = A.KeypointParams(format='xy', remove_invisible=False))

    def prepare_data(self):
        # Download and crop
        CUB2011Parts_dataset(download=True, root=self.data_dir)

    def setup(self, stage=None):
        trainset = CUB2011Parts_dataset(train=True, root=self.data_dir, transform=self.train_transform)
        testset = CUB2011Parts_dataset(train=False, root=self.data_dir, transform=self.test_transform)

        self.num_classes = trainset.num_classes

        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            self.cub_train = trainset
            self.cub_val, _ = random_split(testset, [1000, len(testset) - 1000])

        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            self.cub_test = testset

        # self.dims is returned when you call dm.size()
        self.dims = trainset[0][0].shape

    def train_dataloader(self):
        return DataLoader(self.cub_train, shuffle=True, batch_size=self.batch_size,
                          num_workers=self.num_workers, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.cub_val, shuffle=False, batch_size=self.batch_size,
                          num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.cub_test, shuffle=False, batch_size=self.batch_size,
                          num_workers=self.num_workers)

    def predict_dataloader(self):
        return DataLoader(self.cub_test, shuffle=False, batch_size=self.batch_size,
                          num_workers=self.num_workers)


if __name__ == '__main__':
    data_dir = '../../../data'
    explanation = CUB2011Parts(data_dir=data_dir)
    explanation.prepare_data()

    explanation.setup()
    train_dl = explanation.train_dataloader()
    val_dl = explanation.val_dataloader()
    test_dl = explanation.test_dataloader()
    print(f"Dataset split (train/val/test): {len(train_dl.dataset)}/{len(val_dl.dataset)}/{len(test_dl.dataset)}")
