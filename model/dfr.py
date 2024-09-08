from backbone.feature_extractor import FeatureExtractor
from kornia.filters import gaussian_blur2d
from model.cae import CAE
from torch import cat, matmul, mean, Tensor
from torch.functional import F
from torch.nn import Module, ModuleList


class DFR(Module):
    def __init__(self, feature_extractor: FeatureExtractor, iaffs: ModuleList, cae: CAE, image_size: tuple, eta: tuple, sigma: tuple) -> None:
        super(DFR, self).__init__()
        self._feature_extractor = feature_extractor
        self._iaffs = iaffs
        self._cae = cae
        self._image_size = image_size
        self._eta = eta
        self._sigma = sigma

    def compute_score(self, input: Tensor) -> Tensor:  # (H, W)
        input, output = self._forward_cae(input=input)
        score = (input - output).pow(exponent=2)
        score = F.interpolate(input=score, size=self._image_size, mode='bilinear', align_corners=False)
        score = mean(input=score, dim=1, keepdim=True)
        score = gaussian_blur2d(input=score, kernel_size=(self._sigma[0] * 6 + 1, self._sigma[1] * 6 + 1), sigma=self._sigma)
        return score.squeeze()

    def compute_distribution_loss(self, input: Tensor, distributions: list) -> Tensor:
        features = self._forward_iaff(input=input)
        loss = 0
        for feature, distribution in zip(features, distributions):
            B, C = feature.shape[0:2]
            feature = feature.reshape(shape=(1, C, B, -1))  # (1, C, B, L)
            feature = feature.permute(dims=(3, 2, 1, 0))  # (L, B, C, 1)
            mu = distribution['mu'].reshape(-1, 1, C, 1).repeat_interleave(repeats=B, dim=1)  # (L, B, C, 1)
            sigma = distribution['sigma'].reshape(-1, 1, C, C).repeat_interleave(repeats=B, dim=1)  # (L, B, C, C)
            residual = matmul(input=sigma.inverse(), other=(feature - mu))  # (L, B, C, 1)
            mahalanobis_distance = (matmul(input=residual.permute(dims=(0, 1, 3, 2)), other=residual).abs() + 1e-8).sqrt()
            loss = loss + mean(input=mahalanobis_distance.flatten(), dim=0)
        return loss / len(features)

    def compute_reconstruction_loss(self, input: Tensor) -> Tensor:
        input, output = self._forward_cae(input=input)
        return F.mse_loss(input=input, target=output)

    def _forward_iaff(self, input: Tensor) -> list:
        features = []
        for features_i, iaffs_i in zip(self._feature_extractor(input=input), self._iaffs):
            features_i = [F.adaptive_avg_pool2d(input=F.interpolate(input=feature_i_i, size=self._image_size, mode='bilinear', align_corners=False), output_size=(self._image_size[0] // self._eta[0], self._image_size[1] // self._eta[1])) for feature_i_i in features_i]
            feature_i_i = features_i[0]
            for i, iaff_i_i in enumerate(iaffs_i):
                feature_i_i = iaff_i_i(shallow_feature=feature_i_i, deep_feature=features_i[i + 1])
            features.append(feature_i_i)
        return features

    def _forward_cae(self, input: Tensor) -> tuple:
        input = cat(tensors=self._forward_iaff(input=input), dim=1)
        output = self._cae(input=input)
        return input, output


class DFRPlus(Module):
    def __init__(self, feature_extractor: FeatureExtractor, iaffs: ModuleList, cae: CAE, weight: tuple, image_size: tuple, eta: tuple, sigma: tuple) -> None:
        super(DFRPlus, self).__init__()
        self._feature_extractor = feature_extractor
        self._iaffs = iaffs
        self._cae = cae
        self._weight = weight
        self._image_size = image_size
        self._eta = eta
        self._sigma = sigma

    def compute_score(self, input: Tensor) -> Tensor:  # (H, W)
        input, output = self._forward_cae(input=input)
        score = (input - output).pow(exponent=2)
        score = F.interpolate(input=score, size=self._image_size, mode='bilinear', align_corners=False)
        score = mean(input=score, dim=1, keepdim=True)
        score = gaussian_blur2d(input=score, kernel_size=(self._sigma[0] * 6 + 1, self._sigma[1] * 6 + 1), sigma=self._sigma)
        return score.squeeze()

    def compute_distribution_loss(self, input: Tensor, distributions: list) -> Tensor:
        features = self._forward_iaff(input=input)
        loss = 0
        for feature, distribution in zip(features, distributions):
            B, C = feature.shape[0:2]
            feature = feature.reshape(shape=(1, C, B, -1))  # (1, C, B, L)
            feature = feature.permute(dims=(3, 2, 1, 0))  # (L, B, C, 1)
            mu = distribution['mu'].reshape(-1, 1, C, 1).repeat_interleave(repeats=B, dim=1)  # (L, B, C, 1)
            sigma = distribution['sigma'].reshape(-1, 1, C, C).repeat_interleave(repeats=B, dim=1)  # (L, B, C, C)
            residual = matmul(input=sigma.inverse(), other=(feature - mu))  # (L, B, C, 1)
            mahalanobis_distance = (matmul(input=residual.permute(dims=(0, 1, 3, 2)), other=residual).abs() + 1e-8).sqrt()
            loss = loss + mean(input=mahalanobis_distance.flatten(), dim=0)
        return loss / len(features)

    def compute_reconstruction_loss(self, input: Tensor) -> Tensor:
        input, output = self._forward_cae(input=input)
        return F.mse_loss(input=input, target=output)

    def _forward_iaff(self, input: Tensor) -> list:
        features = []
        for features_i, iaffs_i in zip(self._feature_extractor(input=input), self._iaffs):
            features_i = [F.adaptive_avg_pool2d(input=F.interpolate(input=feature_i_i, size=self._image_size, mode='bilinear', align_corners=False), output_size=(self._image_size[0] // self._eta[0], self._image_size[1] // self._eta[1])) for feature_i_i in features_i]
            feature_i_i = features_i[0]
            for i, iaff_i_i in enumerate(iaffs_i):
                feature_i_i = iaff_i_i(shallow_feature=feature_i_i, deep_feature=features_i[i + 1])
            features.append(feature_i_i)
        return features

    def _forward_cae(self, input: Tensor) -> tuple:
        input = cat(tensors=[feature * weight_i for feature, weight_i in zip(self._forward_iaff(input=input), self._weight)], dim=1)
        output = self._cae(input=input)
        return input, output
