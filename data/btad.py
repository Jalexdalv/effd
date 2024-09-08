from cv2 import COLOR_BGR2RGB, cvtColor, imread, IMREAD_COLOR, IMREAD_GRAYSCALE
from data.base import BaseTrainDataset, BaseTestDataset
from os import listdir
from os.path import join
from torch import Tensor, zeros


class TrainDataset(BaseTrainDataset):
    def __init__(self, category_path: str, batch_size: int, num_workers: int, image_size: tuple, mean: tuple = (0.485, 0.456, 0.406), std: tuple = (0.229, 0.224, 0.225)) -> None:
        super(TrainDataset, self).__init__(batch_size=batch_size, num_workers=num_workers, image_size=image_size, mean=mean, std=std)

        self._image_paths = []
        image_dir_path = join(category_path, 'train', 'ok')
        for image_name in listdir(path=image_dir_path):
            self._image_paths.append(join(image_dir_path, image_name))
        self._image_cache = {}

        self.init_dataloader()

    def __getitem__(self, index: int) -> Tensor:
        if index in self._image_cache:
            return self._image_cache[index]
        else:
            image = self._resize(img=self._normalize_transform(tensor=self._tensor_transform(pic=cvtColor(src=imread(filename=self._image_paths[index], flags=IMREAD_COLOR), code=COLOR_BGR2RGB))))
            self._image_cache[index] = image
            return image

    def __len__(self) -> int:
        return len(self._image_paths)


class TestDataset(BaseTestDataset):
    def __init__(self, category_path: str, num_workers: int, image_size: tuple, mean: tuple = (0.485, 0.456, 0.406), std: tuple = (0.229, 0.224, 0.225)) -> None:
        super(TestDataset, self).__init__(num_workers=num_workers, image_size=image_size, mean=mean, std=std)

        self._image_paths = []
        self._image_names = []
        self._ground_truth_paths = []
        self._defect_categories = []
        image_path = join(category_path, 'test')
        ground_truth_path = join(category_path, 'ground_truth')
        for defect_category in listdir(path=image_path):
            image_defect_category_path = join(image_path, defect_category)
            if defect_category == 'ok':
                for image_name in listdir(path=image_defect_category_path):
                    self._image_paths.append(join(image_defect_category_path, image_name))
                    self._image_names.append(image_name)
                    self._ground_truth_paths.append(None)
                    self._defect_categories.append('ok')
            else:
                ground_truth_defect_category_path = join(ground_truth_path, defect_category)
                for image_name, ground_truth_name in zip(listdir(path=image_defect_category_path), listdir(path=ground_truth_defect_category_path)):
                    self._image_paths.append(join(image_defect_category_path, image_name))
                    self._image_names.append(image_name)
                    self._ground_truth_paths.append(join(ground_truth_defect_category_path, ground_truth_name))
                    self._defect_categories.append(defect_category)

        self.init_dataloader()

    def __getitem__(self, index: int) -> tuple:
        image = self._resize(img=self._normalize_transform(tensor=self._tensor_transform(pic=cvtColor(src=imread(filename=self._image_paths[index], flags=IMREAD_COLOR), code=COLOR_BGR2RGB))))
        ground_truth = zeros(size=(1, *self.image_size)) if self._ground_truth_paths[index] is None else self._resize(img=self._tensor_transform(pic=imread(filename=self._ground_truth_paths[index], flags=IMREAD_GRAYSCALE)))
        return image, ground_truth, self._defect_categories[index], self._image_names[index]

    def __len__(self) -> int:
        return len(self._image_paths)
