from backbone.vgg import Vgg19
from torch import load, Tensor
from torch.nn import Module


class FeatureExtractor(Module):
    def __init__(self, backbone: Module, level_map: dict, path: str) -> None:
        super(FeatureExtractor, self).__init__()
        self._level_map = level_map
        self._backbone = backbone
        self._backbone.load_state_dict(state_dict=load(f=path))
        for parameter in self._backbone.parameters():
            parameter.requires_grad = False
        self.num_channels = []


class Vgg19FeatureExtractor(FeatureExtractor):
    def __init__(self, levels: tuple, pool: str, padding_mode: str) -> None:
        super(Vgg19FeatureExtractor, self).__init__(backbone=Vgg19(pool=pool, padding_mode=padding_mode),
                                                    level_map={'level_1_1': 64, 'level_1_2': 64,
                                                               'level_2_1': 128, 'level_2_2': 128,
                                                               'level_3_1': 256, 'level_3_2': 256, 'level_3_3': 256, 'level_3_4': 256,
                                                               'level_4_1': 512, 'level_4_2': 512, 'level_4_3': 512, 'level_4_4': 512,
                                                               'level_5_1': 512, 'level_5_2': 512, 'level_5_3': 512, 'level_5_4': 512},
                                                    path='pretrain/vgg19-dcbb9e9d.pth')
        self._levels = levels
        self._features = self._backbone.features
        self._levels_1, self._levels_2, self._levels_3, self._levels_4, self._levels_5 = [], [], [], [], []
        self._num_channels_1, self._num_channels_2, self._num_channels_3, self._num_channels_4, self._num_channels_5 = [], [], [], [], []
        for level in levels:
            if str.startswith(level, 'level_1'):
                self._levels_1.append(level)
                self._num_channels_1.append(self._level_map[level])
            elif str.startswith(level, 'level_2'):
                self._levels_2.append(level)
                self._num_channels_2.append(self._level_map[level])
            elif str.startswith(level, 'level_3'):
                self._levels_3.append(level)
                self._num_channels_3.append(self._level_map[level])
            elif str.startswith(level, 'level_4'):
                self._levels_4.append(level)
                self._num_channels_4.append(self._level_map[level])
            elif str.startswith(level, 'level_5'):
                self._levels_5.append(level)
                self._num_channels_5.append(self._level_map[level])
        if self._num_channels_1:
            self.num_channels.append(self._num_channels_1)
        if self._num_channels_2:
            self.num_channels.append(self._num_channels_2)
        if self._num_channels_3:
            self.num_channels.append(self._num_channels_3)
        if self._num_channels_4:
            self.num_channels.append(self._num_channels_4)
        if self._num_channels_5:
            self.num_channels.append(self._num_channels_5)

    def forward(self, input: Tensor) -> list:
        feature_1_1 = self._features[1](input=self._features[0](input=input))
        feature_1_2 = self._features[3](input=self._features[2](input=feature_1_1))
        feature_2_1 = self._features[6](input=self._features[5](input=self._features[4](input=feature_1_2)))
        feature_2_2 = self._features[8](input=self._features[7](input=feature_2_1))
        feature_3_1 = self._features[11](input=self._features[10](input=self._features[9](input=feature_2_2)))
        feature_3_2 = self._features[13](input=self._features[12](input=feature_3_1))
        feature_3_3 = self._features[15](input=self._features[14](input=feature_3_2))
        feature_3_4 = self._features[17](input=self._features[16](input=feature_3_3))
        feature_4_1 = self._features[20](input=self._features[19](input=self._features[18](input=feature_3_4)))
        feature_4_2 = self._features[22](input=self._features[21](input=feature_4_1))
        feature_4_3 = self._features[24](input=self._features[23](input=feature_4_2))
        feature_4_4 = self._features[26](input=self._features[25](input=feature_4_3))
        feature_5_1 = self._features[29](input=self._features[28](input=self._features[27](input=feature_4_4)))
        feature_5_2 = self._features[31](input=self._features[30](input=feature_5_1))
        feature_5_3 = self._features[33](input=self._features[32](input=feature_5_2))
        feature_5_4 = self._features[35](input=self._features[34](input=feature_5_3))
        feature_map = {'level_1_1': feature_1_1, 'level_1_2': feature_1_2,
                       'level_2_1': feature_2_1, 'level_2_2': feature_2_2,
                       'level_3_1': feature_3_1, 'level_3_2': feature_3_2, 'level_3_3': feature_3_3, 'level_3_4': feature_3_4,
                       'level_4_1': feature_4_1, 'level_4_2': feature_4_2, 'level_4_3': feature_4_3, 'level_4_4': feature_4_4,
                       'level_5_1': feature_5_1, 'level_5_2': feature_5_2, 'level_5_3': feature_5_3, 'level_5_4': feature_5_4}
        features, features_1, features_2, features_3, features_4, features_5 = [], [], [], [], [], []
        for level in self._levels_1:
            features_1.append(feature_map[level])
        for level in self._levels_2:
            features_2.append(feature_map[level])
        for level in self._levels_3:
            features_3.append(feature_map[level])
        for level in self._levels_4:
            features_4.append(feature_map[level])
        for level in self._levels_5:
            features_5.append(feature_map[level])
        if features_1:
            features.append(features_1)
        if features_2:
            features.append(features_2)
        if features_3:
            features.append(features_3)
        if features_4:
            features.append(features_4)
        if features_5:
            features.append(features_5)
        return features
