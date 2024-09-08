from cv2 import cvtColor, COLOR_RGB2BGR
from numpy import around, array, ndarray, uint8
from os import makedirs
from os.path import exists
from pickle import dump, load
from torch import device, from_numpy, Tensor
from torch.nn import Module


def create_dir(path: str):
    if not exists(path=path):
        makedirs(name=path)


def get_device(module: Module) -> device:
    return next(module.parameters()).device


def convert_list_to_tensor(list: list) -> Tensor:
    return convert_numpy_to_tensor(ndarray=array(object=list))


def convert_tensor_to_list(tensor: Tensor) -> Tensor:
    return convert_tensor_to_numpy(tensor=tensor).tolist()


def convert_tensor_to_numpy(tensor: Tensor) -> ndarray:
    return tensor.detach().cpu().numpy()


def convert_numpy_to_tensor(ndarray: ndarray) -> Tensor:
    return from_numpy(ndarray)


def convert_tensor_image_to_opencv_image(image: Tensor) -> ndarray:
    image = unormalize_image(image=convert_tensor_to_numpy(tensor=image))
    if len(image.shape) == 3:
        image = image.transpose(1, 2, 0)
        image = cvtColor(src=image, code=COLOR_RGB2BGR)
    return image


def unormalize_image(image: ndarray) -> ndarray:
    return uint8(around(a=image * 255))


def save_to_pickle(object: object, path: str) -> None:
    pkl = open(file=path, mode='wb')
    dump(obj=object, file=pkl)
    pkl.close()


def load_from_pickle(path: str) -> object:
    pkl = open(file=path, mode='rb')
    object = load(file=pkl)
    pkl.close()
    return object


def max_min_normalize(input: ndarray) -> tuple:
    input_max = input.max()
    input_min = input.min()
    output = (input - input_min) / (input_max - input_min)
    return input_max, input_min, output
