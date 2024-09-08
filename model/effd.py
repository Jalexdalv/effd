from backbone.feature_extractor import FeatureExtractor
from data.base import BaseTrainDataset
from sklearn.covariance import LedoitWolf
from torch import bmm, cat, eye, mean, no_grad, stack
from torch.functional import F
from torch.linalg import cholesky
from torch.nn import Module
from utils import convert_numpy_to_tensor, convert_tensor_to_numpy, get_device


class EFFD(Module):
    def __init__(self, feature_extractor: FeatureExtractor, image_size: tuple, eta: tuple) -> None:
        super(EFFD, self).__init__()
        self._feature_extractor = feature_extractor
        self._image_size = image_size
        self._eta = eta

    def get_sample_features(self, train_dataset: BaseTrainDataset) -> list:
        sample_features = [[[] for _ in num_channels_i] for num_channels_i in self._feature_extractor.num_channels]
        with no_grad():
            for input in train_dataset.dataloader:
                for sample_features_i, features_i in zip(sample_features, self._feature_extractor(input=input.to(device=get_device(module=self)))):
                    for sample_features_i_i, feature_i_i in zip(sample_features_i, features_i):
                        sample_features_i_i.append(feature_i_i)
        return sample_features

    def forward(self, sample_features: list) -> list:
        distributions = []
        for sample_features_i in sample_features:
            distribution = {}
            for sample_features_i_i in sample_features_i:
                sample_features_i_i = cat(tensors=[F.adaptive_avg_pool2d(input=sample_feature_i_i, output_size=(self._image_size[0] // self._eta[0], self._image_size[1] // self._eta[1])) for sample_feature_i_i in sample_features_i_i], dim=0)  # (N, c, h, w)
                sample_features_i_i = sample_features_i_i.reshape(shape=(*sample_features_i_i.shape[0:2], -1))  # (N, c, L)
                c = sample_features_i_i.shape[1]
                L = sample_features_i_i.shape[2]
                # mean
                mu = mean(input=sample_features_i_i, dim=0, keepdim=True).permute(dims=(2, 1, 0))  # (L, c, 1)
                # covariance
                # 使用 cholesky 分解，返回下三角矩阵
                sigma = stack(tensors=[cholesky(input=(convert_numpy_to_tensor(ndarray=LedoitWolf().fit(X=convert_tensor_to_numpy(tensor=sample_features_i_i[:, :, l])).covariance_) + 0.01 * eye(n=c)).to(device=sample_features_i_i.device)) for l in range(L)], dim=0)  # (L, c, c)
                # fuse
                P = bmm(input=distribution['sigma'], mat2=(distribution['sigma'] + sigma).inverse()) if 'sigma' in distribution else None  # (L, c, c)
                if 'mu' in distribution:
                    distribution['mu'] = (distribution['mu'] + bmm(input=P, mat2=mu - distribution['mu']))  # (L, c, 1)
                else:
                    distribution['mu'] = mu  # (L, c, 1)
                if 'sigma' in distribution:
                    distribution['sigma'] = distribution['sigma'] - bmm(input=P, mat2=distribution['sigma'])  # (L, c, c)
                else:
                    distribution['sigma'] = sigma  # (L, c, c)
            distributions.append(distribution)
        return distributions
