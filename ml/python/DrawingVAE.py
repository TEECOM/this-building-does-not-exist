import torch
import torch.nn as nn
import torch.nn.functional as nnf
from torchviz import make_dot

class Container():
    def __init__(self, x, x_dict):
        self.x = x
        self.x_dict = x_dict
    

class Encoder:
    class D4Linear(nn.Linear):
        def __init__(self, in_features, out_features, bias=True):
            super().__init__(in_features, out_features, bias)
        
        def forward(self, x):
            nx, cx, hx, wx = x.shape

            x = super().forward(x.view(nx, cx))

            return x.view(nx, cx, hx, wx)


    class ConvBlock(nn.Module):
        def __init__(self, in_channels, out_channels, block_name):
            super().__init__()

            self.block_name = block_name

            self.convolution = nn.Conv2d(
                in_channels,
                out_channels,
                4,
                2,
                1,
                bias=False
                )
            
            self.batchnorm = nn.BatchNorm2d(out_channels)

        def forward(self, container):

            x = self.convolution(container.x)
            x = self.batchnorm(x)

            x =  nnf.leaky_relu(x, negative_slope=0.1)

            if container.x_dict is not None:
                container.x_dict[self.block_name] = x

            return Container(x, container.x_dict)

    class DrawingEncoder(nn.Module):
        def __init__(self, input_channels=3, latent_dim=512):
            super().__init__()

            self.main = nn.Sequential()

            self.layer_params = [
                (input_channels, 8), #256
                (8, 16), #128
                (16, 32), #64
                (32, 64), #32
                (64, 128), #16
                (128, 256), #8
                (256, 512), #4
                (512, 512), #2
                (512, latent_dim) #1
            ]

            for n, params in enumerate(self.layer_params):
                name = "cb{}".format(n)
                layer = Encoder.ConvBlock(*(*params, name))

                self.main.add_module(name, layer)

            self.fc_mu = Encoder.D4Linear(latent_dim, latent_dim)
            self.fc_sigma = Encoder.D4Linear(latent_dim, latent_dim)

        def forward(self, container):
            
            container = self.main(container)

            x_mu = nnf.leaky_relu(self.fc_mu(container.x))
            x_sigma = nnf.leaky_relu(self.fc_sigma(container.x))

            return Container((x_mu, x_sigma), container.x_dict)


class Decoder:
    class ConvBlock(nn.Module):
        def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1, block_name=""):
            super().__init__()

            self.block_name = block_name

            self.convolution = nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False
            )

            self.batchnorm = nn.BatchNorm2d(out_channels)

            self.last_layer = False

        def forward(self, container):
            x = container.x
            if container.x_dict is not None:
                x += container.x_dict[self.block_name]
            
            x = self.convolution(x)
            x = self.batchnorm(x)

            if not self.last_layer:
                return Container(nnf.leaky_relu(x, negative_slope=0.1), container.x_dict)
            else:
                return Container(torch.tanh(x), container.x_dict)
    
    class DrawingDecoder(nn.Module):
        def __init__(self, output_channels=3, latent_dim=512):
            super().__init__()

            self.fc = Encoder.D4Linear(latent_dim, latent_dim)

            self.main = nn.Sequential()

            self.layer_params = [
                (latent_dim, 512), #2
                (512, 512), #4
                (512, 256), #8
                (256, 128), #16
                (128, 64), #32
                (64, 32), #64
                (32, 16), #128
                (16, 8), #256
                (8, output_channels) #512
            ]

            for n, params in enumerate(self.layer_params):

                name = "cb{}".format((len(self.layer_params) - 1) - n)
                layer = Decoder.ConvBlock(*params, block_name=name)

                if n == len(self.layer_params) - 1:
                    layer.last_layer = True

                self.main.add_module(name, layer)
        

        def forward(self, container):
            x_mu, x_sigma = container.x

            nx, cx, hx, wx = x_mu.shape

            norm_sample = torch.randn(nx, cx, hx, wx).cuda()

            x = (norm_sample * x_sigma) + x_mu

            x = nnf.leaky_relu(self.fc(x))
            container = self.main(Container(x, container.x_dict))
            return container.x


if __name__ == "__main__":
    de = Encoder.DrawingEncoder().cuda()
    dd = Decoder.DrawingDecoder().cuda()

    x = torch.randn(2, 3, 512, 512).cuda()
    x.requires_grad = True

    enc = de(Container(x, {}))
    dec = dd(enc)

    dec.mean().backward()

    print(de)
    print(dd)

    make_dot(dec).render(r"C:\Users\tyler.kvochick\Desktop\vae")