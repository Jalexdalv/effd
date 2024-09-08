from torch import ones_like, Tensor
from torch.functional import F
from torch.nn import AdaptiveAvgPool2d, BatchNorm2d, Conv2d, Module, ReLU, Sequential
from torch.nn.init import constant_, kaiming_normal_


class MSCAM(Module):
    def __init__(self, in_channels: int, gamma: int) -> None:
        super(MSCAM, self).__init__()
        out_channels = in_channels // gamma
        self._global_attention = Sequential(
            AdaptiveAvgPool2d(output_size=(1, 1)),
            Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=False),
            BatchNorm2d(num_features=out_channels),
            ReLU(inplace=True),
            Conv2d(in_channels=out_channels, out_channels=in_channels, kernel_size=1, bias=False),
            BatchNorm2d(num_features=in_channels)
        )
        self._local_attention = Sequential(
            Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=False),
            BatchNorm2d(num_features=out_channels),
            ReLU(inplace=True),
            Conv2d(in_channels=out_channels, out_channels=in_channels, kernel_size=1, bias=False),
            BatchNorm2d(num_features=in_channels)
        )
        for module in self.modules():
            if isinstance(module, Conv2d):
                kaiming_normal_(tensor=module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    constant_(tensor=module.bias, val=0)
            elif isinstance(module, BatchNorm2d):
                constant_(tensor=module.weight, val=1)
                constant_(tensor=module.bias, val=0)

    def forward(self, input: Tensor) -> Tensor:
        return F.sigmoid(input=self._global_attention(input=input) + self._local_attention(input=input))


class MSCAMPlus(Module):
    def __init__(self, in_channels: int, gamma: int) -> None:
        super(MSCAMPlus, self).__init__()
        out_channels = in_channels // gamma
        self._global_attention = Sequential(
            AdaptiveAvgPool2d(output_size=(1, 1)),
            Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=False),
            BatchNorm2d(num_features=out_channels),
            ReLU(inplace=True),
            Conv2d(in_channels=out_channels, out_channels=in_channels, kernel_size=1, bias=False),
            BatchNorm2d(num_features=in_channels)
        )
        self._local_attention_1 = Sequential(
            Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=False),
            BatchNorm2d(num_features=out_channels),
            ReLU(inplace=True),
            Conv2d(in_channels=out_channels, out_channels=in_channels, kernel_size=1, bias=False),
            BatchNorm2d(num_features=in_channels)
        )
        self._local_attention_2 = Sequential(
            Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, bias=False),
            BatchNorm2d(num_features=out_channels),
            ReLU(inplace=True),
            Conv2d(in_channels=out_channels, out_channels=in_channels, kernel_size=3, padding=1, bias=False),
            BatchNorm2d(num_features=in_channels)
        )
        self._local_attention_3 = Sequential(
            Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=5, padding=2, bias=False),
            BatchNorm2d(num_features=out_channels),
            ReLU(inplace=True),
            Conv2d(in_channels=out_channels, out_channels=in_channels, kernel_size=5, padding=2, bias=False),
            BatchNorm2d(num_features=in_channels)
        )
        for module in self.modules():
            if isinstance(module, Conv2d):
                kaiming_normal_(tensor=module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    constant_(tensor=module.bias, val=0)
            elif isinstance(module, BatchNorm2d):
                constant_(tensor=module.weight, val=1)
                constant_(tensor=module.bias, val=0)

    def forward(self, input: Tensor) -> Tensor:
        return F.sigmoid(input=self._global_attention(input=input) + (self._local_attention_1(input=input) + self._local_attention_2(input=input) + self._local_attention_3(input=input)))


class IAFF(Module):
    def __init__(self, in_channels: int, gamma: int) -> None:
        super(IAFF, self).__init__()
        self._mscam_1 = MSCAM(in_channels=in_channels, gamma=gamma)
        self._mscam_2 = MSCAM(in_channels=in_channels, gamma=gamma)

    def forward(self, shallow_feature: Tensor, deep_feature: Tensor) -> Tensor:
        output = shallow_feature + deep_feature
        output = self._mscam_1(input=output)
        output = shallow_feature * output + deep_feature * (ones_like(input=output) - output)
        output = self._mscam_2(input=output)
        output = shallow_feature * output + deep_feature * (ones_like(input=output) - output)
        return output


class IAFFPlus(Module):
    def __init__(self, in_channels: int, gamma: int) -> None:
        super(IAFFPlus, self).__init__()
        self._mscam_1 = MSCAMPlus(in_channels=in_channels, gamma=gamma)
        self._mscam_2 = MSCAMPlus(in_channels=in_channels, gamma=gamma)

    def forward(self, shallow_feature: Tensor, deep_feature: Tensor) -> Tensor:
        output = shallow_feature + deep_feature
        output = self._mscam_1(input=output)
        output = shallow_feature * output + deep_feature * (ones_like(input=output) - output)
        output = self._mscam_2(input=output)
        output = shallow_feature * output + deep_feature * (ones_like(input=output) - output)
        return output
