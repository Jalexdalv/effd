from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Normalize, Resize, ToTensor
from torchvision.transforms.functional import InterpolationMode


class BaseTrainDataset(Dataset):
    def __init__(self, batch_size: int, num_workers: int, image_size: tuple, mean: tuple = (0.485, 0.456, 0.406), std: tuple = (0.229, 0.224, 0.225)) -> None:
        super(BaseTrainDataset, self).__init__()
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._resize = Resize(size=image_size, interpolation=InterpolationMode.NEAREST)
        self._tensor_transform = ToTensor()
        self._normalize_transform = Normalize(mean=mean, std=std)
        self.image_size = image_size
        self.mean = mean
        self.std = std
        self.dataloader = None

    def init_dataloader(self):
        self.dataloader = DataLoader(dataset=self, batch_size=self._batch_size, shuffle=True, num_workers=self._num_workers, drop_last=True)

    def __getitem__(self, index: int) -> Tensor:
        pass

    def __len__(self) -> int:
        pass


class BaseTestDataset(Dataset):
    def __init__(self, num_workers: int, image_size: tuple, mean: tuple = (0.485, 0.456, 0.406), std: tuple = (0.229, 0.224, 0.225)) -> None:
        super(BaseTestDataset, self).__init__()
        self._num_workers = num_workers
        self._resize = Resize(size=image_size, interpolation=InterpolationMode.NEAREST)
        self._tensor_transform = ToTensor()
        self._normalize_transform = Normalize(mean=mean, std=std)
        self.image_size = image_size
        self.mean = mean
        self.std = std
        self.dataloader = None

    def init_dataloader(self):
        self.dataloader = DataLoader(dataset=self, batch_size=1, shuffle=False, num_workers=self._num_workers)

    def __getitem__(self, index: int) -> tuple:
        pass

    def __len__(self) -> int:
        pass
