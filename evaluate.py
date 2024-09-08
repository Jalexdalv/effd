from cv2 import applyColorMap, CHAIN_APPROX_SIMPLE, COLORMAP_JET, drawContours, findContours, imwrite, normalize, NORM_MINMAX, RETR_TREE
from data.base import BaseTrainDataset, BaseTestDataset
from numpy import array
from os.path import join
from sklearn.metrics import roc_auc_score
from torch import no_grad, sort, stack
from torch.nn import Module
from utils import convert_tensor_to_numpy, convert_tensor_image_to_opencv_image, create_dir, get_device, unormalize_image


def _binary_score(model: Module, test_dataset: BaseTestDataset) -> tuple:
    scores, ground_truths = [], []
    with no_grad():
        for input, ground_truth, _, _ in test_dataset.dataloader:
            scores.append(convert_tensor_to_numpy(tensor=model.compute_score(input=input.to(device=get_device(module=model)))))
            ground_truths.append(convert_tensor_to_numpy(tensor=ground_truth.squeeze()))
    scores = array(object=scores)
    ground_truths = array(object=ground_truths)
    ground_truths[ground_truths <= 0.5] = 0
    ground_truths[ground_truths > 0.5] = 1
    return scores, ground_truths


def compute_auc_roc(model: Module, test_dataset: BaseTestDataset) -> float:
    scores, ground_truths = _binary_score(model=model, test_dataset=test_dataset)
    auc_roc = roc_auc_score(y_true=ground_truths.ravel(), y_score=scores.ravel())
    print('auc-roc：{}'.format(auc_roc))
    return auc_roc


def compute_threshold(model: Module, train_dataset: BaseTrainDataset, expect_fprs: tuple) -> dict:
    thresholds = {}
    with no_grad():
        scores = stack(tensors=[model.compute_score(input=input.to(device=get_device(module=model))) for input in train_dataset.dataloader], dim=0)
    for expect_fpr in expect_fprs:
        threshold = sort(input=scores.flatten(), descending=True)[0][int(scores.shape[0] * scores.shape[1] * scores.shape[2] * expect_fpr)].item()
        thresholds[expect_fpr] = threshold
        print('expect_fpr: {}%  threshold: {}'.format(expect_fpr * 100, threshold))
    return thresholds


def segment(model: Module, test_dataset: BaseTestDataset, thresholds: dict, result_path: str):
    with no_grad():
        for input, ground_truth, defect_category, name in test_dataset.dataloader:
            # (H, W)
            score = convert_tensor_to_numpy(tensor=model.compute_score(input=input.to(device=get_device(module=model))))
            # (3, H, W)
            input = input.squeeze()
            input[0] = input[0] * test_dataset.std[0] + test_dataset.mean[0]
            input[1] = input[1] * test_dataset.std[1] + test_dataset.mean[1]
            input[2] = input[2] * test_dataset.std[2] + test_dataset.mean[2]
            # (H, W, 3)
            input = convert_tensor_image_to_opencv_image(image=input)
            # (H, W)
            ground_truth = ground_truth.squeeze()
            ground_truth = convert_tensor_image_to_opencv_image(image=ground_truth)

            path = join(result_path, defect_category[0], name[0])
            create_dir(path=path)
            imwrite(filename=join(path, 'test.png'), img=input)
            imwrite(filename=join(path, 'ground_truth.png'), img=ground_truth)

            heat_map = normalize(src=unormalize_image(image=score), dst=None, alpha=0, beta=255, norm_type=NORM_MINMAX)
            heat_map = applyColorMap(src=heat_map, colormap=COLORMAP_JET) * 0.7 + input * 0.5
            heat_map_path = join(path, 'heat_map.png')
            imwrite(filename=heat_map_path, img=heat_map)
            print("success output heat_map：{}".format(heat_map_path))

            for expect_fpr in thresholds.keys():
                segment_result = input.copy()
                segment_result[:, :, 0][score >= thresholds[expect_fpr]] = 124
                segment_result[:, :, 1][score >= thresholds[expect_fpr]] = 252
                segment_result[:, :, 2][score >= thresholds[expect_fpr]] = 0
                contours = findContours(image=ground_truth, mode=RETR_TREE, method=CHAIN_APPROX_SIMPLE)[0]
                drawContours(image=segment_result, contours=contours, contourIdx=-1, color=(0, 0, 255), thickness=2)
                segment_path = join(path, "fpr-{}.png".format(expect_fpr))
                imwrite(filename=segment_path, img=segment_result)
                print("success output segment：{}".format(segment_path))
