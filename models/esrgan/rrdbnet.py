import torch
import torch.nn as nn


class DenseResidualBlock(nn.Module):

    def __init__(self, channels=64, growth_channels=32):
        super().__init__()

        self.conv1 = nn.Conv2d(
            channels,
            growth_channels,
            3,
            1,
            1
        )

        self.conv2 = nn.Conv2d(
            channels + growth_channels,
            growth_channels,
            3,
            1,
            1
        )

        self.conv3 = nn.Conv2d(
            channels + growth_channels * 2,
            growth_channels,
            3,
            1,
            1
        )

        self.conv4 = nn.Conv2d(
            channels + growth_channels * 3,
            growth_channels,
            3,
            1,
            1
        )

        self.conv5 = nn.Conv2d(
            channels + growth_channels * 4,
            channels,
            3,
            1,
            1
        )

        self.lrelu = nn.LeakyReLU(
            negative_slope=0.2,
            inplace=True
        )

    def forward(self, x):

        x1 = self.lrelu(self.conv1(x))

        x2 = self.lrelu(
            self.conv2(torch.cat((x, x1), 1))
        )

        x3 = self.lrelu(
            self.conv3(torch.cat((x, x1, x2), 1))
        )

        x4 = self.lrelu(
            self.conv4(torch.cat((x, x1, x2, x3), 1))
        )

        x5 = self.conv5(
            torch.cat((x, x1, x2, x3, x4), 1)
        )

        return x + 0.2 * x5


class RRDB(nn.Module):

    def __init__(self, channels=64, growth_channels=32):
        super().__init__()

        self.block1 = DenseResidualBlock(
            channels,
            growth_channels
        )

        self.block2 = DenseResidualBlock(
            channels,
            growth_channels
        )

        self.block3 = DenseResidualBlock(
            channels,
            growth_channels
        )

    def forward(self, x):

        out = self.block1(x)
        out = self.block2(out)
        out = self.block3(out)

        return x + 0.2 * out


class RRDBNet(nn.Module):

    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        features=64,
        growth_channels=32,
        num_blocks=6,
        scale=4
    ):
        super().__init__()

        self.scale = scale

        self.conv_first = nn.Conv2d(
            in_channels,
            features,
            3,
            1,
            1
        )

        self.body = nn.Sequential(
            *[
                RRDB(
                    features,
                    growth_channels
                )
                for _ in range(num_blocks)
            ]
        )

        self.trunk_conv = nn.Conv2d(
            features,
            features,
            3,
            1,
            1
        )

        self.upconv1 = nn.Conv2d(
            features,
            features,
            3,
            1,
            1
        )

        self.upconv2 = nn.Conv2d(
            features,
            features,
            3,
            1,
            1
        )

        self.hr_conv = nn.Conv2d(
            features,
            features,
            3,
            1,
            1
        )

        self.conv_last = nn.Conv2d(
            features,
            out_channels,
            3,
            1,
            1
        )

        self.lrelu = nn.LeakyReLU(
            negative_slope=0.2,
            inplace=True
        )

    def forward(self, x):

        feat = self.conv_first(x)

        body = self.body(feat)

        body = self.trunk_conv(body)

        feat = feat + body

        feat = self.lrelu(
            self.upconv1(
                nn.functional.interpolate(
                    feat,
                    scale_factor=2,
                    mode="nearest"
                )
            )
        )

        feat = self.lrelu(
            self.upconv2(
                nn.functional.interpolate(
                    feat,
                    scale_factor=2,
                    mode="nearest"
                )
            )
        )

        out = self.lrelu(
            self.hr_conv(feat)
        )

        out = self.conv_last(out)

        return out