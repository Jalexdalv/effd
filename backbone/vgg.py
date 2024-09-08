from torch import flatten, Tensor
from torch.nn import AdaptiveAvgPool2d, AvgPool2d, Conv2d, Dropout, Linear, MaxPool2d, Module, ReLU, Sequential


class Vgg19(Module):
    def __init__(self, pool: str, padding_mode: str) -> None:
        super(Vgg19, self).__init__()
        assert pool == 'avgpool' or pool == 'maxpool'
        assert padding_mode == 'zeros' or padding_mode == 'reflect' or padding_mode == 'replicate' or padding_mode == 'circular'
        self.features = Sequential(
            Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, padding_mode=padding_mode),
            ReLU(inplace=True),
            Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, padding_mode=padding_mode),
            ReLU(inplace=True),
            AvgPool2d(kernel_size=2, stride=2) if pool == 'avgpool' else MaxPool2d(kernel_size=2, stride=2),
            Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, padding_mode=padding_mode),
            ReLU(inplace=True),
            Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, padding_mode=padding_mode),
            ReLU(inplace=True),
            AvgPool2d(kernel_size=2, stride=2) if pool == 'avgpool' else MaxPool2d(kernel_size=2, stride=2),
            Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, padding_mode=padding_mode),
            ReLU(inplace=True),
            Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, padding_mode=padding_mode),
            ReLU(inplace=True),
            Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, padding_mode=padding_mode),
            ReLU(inplace=True),
            Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, padding_mode=padding_mode),
            ReLU(inplace=True),
            AvgPool2d(kernel_size=2, stride=2) if pool == 'avgpool' else MaxPool2d(kernel_size=2, stride=2),
            Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1, padding_mode=padding_mode),
            ReLU(inplace=True),
            Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, padding_mode=padding_mode),
            ReLU(inplace=True),
            Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, padding_mode=padding_mode),
            ReLU(inplace=True),
            Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, padding_mode=padding_mode),
            ReLU(inplace=True),
            AvgPool2d(kernel_size=2, stride=2) if pool == 'avgpool' else MaxPool2d(kernel_size=2, stride=2),
            Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, padding_mode=padding_mode),
            ReLU(inplace=True),
            Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, padding_mode=padding_mode),
            ReLU(inplace=True),
            Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, padding_mode=padding_mode),
            ReLU(inplace=True),
            Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, padding_mode=padding_mode),
            ReLU(inplace=True),
            AvgPool2d(kernel_size=2, stride=2) if pool == 'avgpool' else MaxPool2d(kernel_size=2, stride=2)
        )
        self.avgpool = AdaptiveAvgPool2d(output_size=7)
        self.classifier = Sequential(
            Linear(in_features=25088, out_features=4096),
            ReLU(inplace=True),
            Dropout(),
            Linear(in_features=4096, out_features=4096),
            ReLU(inplace=True),
            Dropout(),
            Linear(in_features=4096, out_features=1000)
        )

    def forward(self, input: Tensor) -> Tensor:
        return self.classifier(input=flatten(input=self.avgpool(input=self.features(input=input)), start_dim=1))
