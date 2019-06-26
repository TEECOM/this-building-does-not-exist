import torch
import torch.nn as nn
import torch.nn.functional as nnf

class Linear4D(nn.Linear):
    def __init__(self, in_features, out_features):
        super().__init__(in_features, out_features, bias=False)
    
    def forward(self, x):
        nx, cx, hx, wx = x.shape

        x = super().forward(x.view(nx, cx))

        return x.view(nx, self.out_features, 1, 1)

class Discriminator:
    class ConvBlock(nn.Module):
        def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1):
            super().__init__()

            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
            self.bn = nn.BatchNorm2d(out_channels)

        def forward(self, x):
            x = self.conv(x)
            x = self.bn(x)

            return nnf.leaky_relu(x, negative_slope=0.2)
            

    class Discriminator(nn.Module):
        def __init__(self, conv_block_class, in_channels=3, feature_coeff=8, encoding_dim=128, out_categories=2):
            super().__init__()

            self.main = nn.Sequential()

            fc = feature_coeff

            self.layer_params = [
                (3, fc * 8),
                (fc * 8, fc * 8),
                (fc * 8, fc * 8),
                (fc * 8, fc * 8),
                (fc * 8, fc * 16),
                (fc * 16, fc * 32),
                (fc * 32, fc * 64),
                (fc * 64, fc * 64),
                (fc * 64, encoding_dim, 2, 2, 0)
            ]

            for num, params in self.layer_params:
                name = "convblock{}".format(num)
                layer = conv_block_class(*params)

                self.main.add_module(name, layer)
            
            self.encoding = Linear4D(encoding_dim, encoding_dim)
            self.classifier = Linear4D(encoding_dim, out_categories)


if __name__ == "__main__":

    l = Linear4D(3, 8)

    x = torch.randn(8, 3, 1, 1)

    y = l(x)

    print(y.shape)