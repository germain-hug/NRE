import torch.nn as nn

from .inception import BaseInception3


class CoarseModel(BaseInception3):
    """The coarse Inception3-based CNN model."""

    def __init__(self, init_weights=False):
        super(CoarseModel, self).__init__()

        # The list of conv blocks used in the backbones
        self.Conv2d_1a_3x3 = self.conv_block(3, 32, kernel_size=3, stride=2, padding=1)
        self.Conv2d_2a_3x3 = self.conv_block(32, 32, kernel_size=3, padding=1)
        self.Conv2d_2b_3x3 = self.conv_block(32, 64, kernel_size=3, padding=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.Conv2d_3b_1x1 = self.conv_block(64, 80, kernel_size=1)
        self.Conv2d_4a_3x3 = self.conv_block(80, 192, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.Mixed_5b = self.inception_a(192, pool_features=32)
        self.Mixed_5c = self.inception_a(256, pool_features=64)
        self.Mixed_5d = self.inception_a(288, pool_features=64)
        self.Mixed_6a = self.inception_b(288)
        self.Mixed_6b = self.inception_c(768, channels_7x7=128)
        self.Mixed_6c = self.inception_c(768, channels_7x7=160)
        self.Mixed_6d = self.inception_c(768, channels_7x7=160)
        self.Mixed_6e = self.inception_c(768, channels_7x7=192)
        self.Mixed_7a = self.inception_d(768)

        if init_weights:
            self.init_weights()


class FineModel(BaseInception3):
    """The fine Inception3-based CNN model."""

    def __init__(self, init_weights=False):
        super(FineModel, self).__init__()

        # The list of conv blocks used in the backbones
        self.Conv2d_1a_3x3 = self.conv_block(3, 32, kernel_size=3, stride=2, padding=1)
        self.Conv2d_2a_3x3 = self.conv_block(32, 32, kernel_size=3, padding=1)
        self.Conv2d_2b_3x3 = self.conv_block(32, 64, kernel_size=3, padding=1)
        self.Conv2d_3b_1x1 = self.conv_block(64, 80, kernel_size=3, padding=1)
        self.Conv2d_4a_3x3 = self.conv_block(80, 192, kernel_size=3, padding=1)
        self.Mixed_5b = self.inception_a(192, pool_features=32)
        self.Mixed_5c = self.inception_a(256, pool_features=64)
        self.Mixed_5d = self.inception_a(288, pool_features=64)
        self.Mixed_6a = self.inception_b(288, padding=0)
        self.Mixed_6b = self.inception_c(768, channels_7x7=128)
        self.Mixed_6c = self.inception_c(768, channels_7x7=160)
        self.Mixed_6d = self.inception_c(768, channels_7x7=160)
        self.Mixed_6e = self.inception_c(768, channels_7x7=192)
        self.Mixed_7a = self.inception_d(768, version="fine")
        self.Mixed_7b = self.inception_e(1280)
        self.Mixed_7c = self.inception_e(2048)

        if init_weights:
            self.init_weights()
