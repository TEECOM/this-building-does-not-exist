import torch
import torch.nn as nn
import torch.nn.functional as nnf
from torchviz import make_dot
import matplotlib.pyplot as plt


class MappingNetwork(nn.Module):
    """
    See https://arxiv.org/pdf/1812.04948.pdf section 2 for definition
    """
    def __init__(self, layer_count=8, latent_dim=512):
        super(MappingNetwork, self).__init__()

        self.main = nn.Sequential()

        for layer_number in range(layer_count):
            layer_name = "fc{}".format(layer_number)
            layer = nn.Linear(latent_dim, latent_dim)
            self.main.add_module(layer_name, layer)
    
    def forward(self, x):
        n, c, h, w = x.shape
        x = x.reshape(n, c)
        x = self.main(x)

        return x.reshape(n, c, h, w)


class TensorMap:
    def __init__(self, x, w):
        self.x = x
        self.w = w
    
    def __repr__(self):
        return "x: {} w: {}".format(self.x.shape, self.w.shape)


class Generator:
    """
    See https://arxiv.org/pdf/1812.04948.pdf section 2 for definition
    """
    class A(nn.Module):
        def __init__(self, num_features, w_dim=512):
            super(Generator.A, self).__init__()

            self.w_dim = w_dim
            self.num_features = num_features

            self.affine0 = nn.Linear(w_dim, num_features)
            self.affine1 = nn.Linear(w_dim, num_features)
        
        def forward(self, w):
            n, c, h, width = w.shape

            w = w.reshape(n, self.w_dim)

            y_s = self.affine0(w).reshape(n, self.num_features, h, width)
            y_b = self.affine1(w).reshape(n, self.num_features, h, width)
            return (y_s, y_b)


    class B(nn.Module):
        def __init__(self, height, width, num_features):
            super(Generator.B, self).__init__()
            self.width = width
            self.height = height
            self.num_features = num_features
            
            self.noise_image = torch.randn(1, 1, height, width)
            
            self.scaling_factors = torch.nn.Parameter(data=torch.randn(1, num_features, 1, 1), requires_grad=True)
            
        def forward(self):
            return self.scaling_factors.expand(1, -1, self.height, self.width) * self.noise_image


    class AdaIN(nn.Module):
        def __init__(self):
            super(Generator.AdaIN, self).__init__()
            
        def forward(self, x, y):
            # y is coming from an A block
            # x is the generated images
            n, c, h, w = x.shape
            mu_x    = x.mean(dim=(0, 2, 3)).reshape(1, -1, 1, 1)
            sigma_x = x.std(dim=(0, 2, 3)).reshape(1, -1, 1, 1)
            
            normed_x = (x - mu_x) / sigma_x
            
            y_s, y_b = y
            
            return (y_s * x) + y_b


    class PixelNorm(nn.Module):
        """
        See https://arxiv.org/pdf/1710.10196.pdf section 4.2 for definition
        """
        def __init__(self):
            super(Generator.PixelNorm, self).__init__()
            self.epsilon = 10 ** -8
            
        def forward(self, x):
            n, c, h, w = x.shape
            d = x.pow(2)
            d = d.sum(dim=(1)) + self.epsilon
            d = d.mul(1 / c)
            d = d.sqrt()
            d = d.unsqueeze(1)

            return x / d


    class ConvBlock(nn.Module):
        def __init__(self, in_channels, out_channels):
            super(Generator.ConvBlock, self).__init__()

            self.in_channels = in_channels
            self.out_channels = out_channels
            
            self.conv = nn.Conv2d(in_channels, out_channels, (3, 3), 1, 1)
            self.conv.weight.data.normal_(0, 1)
            self.conv.bias.data.fill_(0)
            
            self.norm = Generator.PixelNorm()
            self.act = nn.LeakyReLU(negative_slope=0.2)
        
        def forward(self, x):
            x = self.conv(x)
            x = self.norm(x)
            x = self.act(x)
            
            return x


    class SynthesisBlock(nn.Module):
        def __init__(self, in_channels, out_channels, height, width, w_dim=512):
            super(Generator.SynthesisBlock, self).__init__()
            
            self.upsample = nn.UpsamplingBilinear2d((height, width))
            self.conv0 =    Generator.ConvBlock(in_channels, out_channels)
            self.b0 =       Generator.B(height, width, out_channels)
            self.a0 =       Generator.A(out_channels, w_dim=w_dim)
            self.adain0 =   Generator.AdaIN()
            
            self.conv1 =    Generator.ConvBlock(out_channels, out_channels)
            self.b1 =       Generator.B(height, width, out_channels)
            self.a1 =       Generator.A(out_channels, w_dim=w_dim)
            self.adain1 =   Generator.AdaIN()
        
        def forward(self, tensor_map):

            x = tensor_map.x
            w = tensor_map.w
            
            x = self.upsample(x)
            x = self.conv0(x)
            x = x + self.b0()
            y = self.a0(w)
            x = self.adain0(x, y)
            
            x = self.conv1(x)
            x = x + self.b1()
            y = self.a1(w)
            x = self.adain1(x, y)
            
            return TensorMap(x, w)


    class InputBlock(nn.Module):
        def __init__(self, in_channels, out_channels, height, width, w_dim=512):
            super(Generator.InputBlock, self).__init__()
            
            self.constant = torch.nn.Parameter(data=torch.randn(1, in_channels, height, width), requires_grad=True)
            self.conv0 =    Generator.ConvBlock(in_channels, out_channels)
            self.b0 =       Generator.B(height, width, out_channels)
            self.a0 =       Generator.A(out_channels, w_dim=w_dim)
            self.adain0 =   Generator.AdaIN()
            
            self.b1 =       Generator.B(height, width, out_channels)
            self.a1 =       Generator.A(out_channels, w_dim=w_dim)
            self.adain1 =   Generator.AdaIN()
        
        def forward(self, batch_size, tensor_map):

            w = tensor_map.w

            x = self.conv0(self.constant.expand(batch_size, -1, -1, -1))
            x = x + self.b0()
            y = self.a0(w)
            x = self.adain0(x, y)

            x = x + self.b1()
            y = self.a1(w)
            x = self.adain1(x, y)
            
            return TensorMap(x, w)


    class OutputBlock(nn.Module):
        def __init__(self, input_channels):
            super(Generator.OutputBlock, self).__init__()
            
            self.to_rgb = nn.Conv2d(input_channels, 3, (1, 1))
        
        def forward(self, tensor_map):
            x = tensor_map.x
            w = tensor_map.w
            
            x = self.to_rgb(x)
            
            return TensorMap(x, w)


    class FadeIn(nn.Module):
        """
        See https://arxiv.org/pdf/1710.10196.pdf section 3 for definition
        """
        def __init__(self):
            super(Generator.FadeIn, self).__init__()

            self.alpha = 0

            self.main = nn.Sequential()

        def add_fade_layer(self, layer):
            self.main.add_module("fade_in", layer)
            self.main.add_module("output", Generator.OutputBlock(layer.conv1.out_channels))
        
        def clear(self):
            del(self.main.fade_in)
            del(self.main.output)
        
        def forward(self, tensor_map):
            return self.main(tensor_map)


    class StyleGenerator(nn.Module):
        def __init__(self, n_batches, input_layer=None, layer_params=None, w_dim=512):
            super(Generator.StyleGenerator, self).__init__()

            if input_layer == None:
                input_layer = Generator.InputBlock(512, 512, 4, 4, w_dim=w_dim)
            
            self.n_batches = n_batches
            
            self.init_alpha()
            
            self.input = input_layer

            self.main = nn.Sequential()

            self.fade_in = Generator.FadeIn()


            if layer_params == None:
                layer_params = [
                    (512, 512,    8,    8, w_dim),
                    (512, 512,   16,   16, w_dim),
                    (512, 512,   32,   32, w_dim),
                    (512, 256,   64,   64, w_dim),
                    (256, 128,  128,  128, w_dim),
                    (128,  64,  256,  256, w_dim),
                    ( 64,  32,  512,  512, w_dim),
                    ( 32,  16, 1024, 1024, w_dim),
                ]
            
            self.layer_params = layer_params
            
            self.main_layer_count = len(self.layer_params)

            self.output = Generator.OutputBlock(input_layer.conv0.out_channels)

            self.upsample = None
        
        def init_alpha(self):
            self.alpha_step = (1 / self.n_batches)
            
            self.alpha_progression = list(torch.arange(self.alpha_step, 1 + self.alpha_step, self.alpha_step, dtype=torch.float32).flip(0))
            
            self.alpha = 0
        
        def step_alpha(self):
            if len(self.alpha_progression) == 0:
                return
            self.alpha = self.alpha_progression.pop(0)


        def step_training_progression(self, epoch_number):
            """
            This is meant to be called once per epoch to add the next layer in the progression.
            """
            if len(self.layer_params) == 0:
                print("All blocks are added")
                pre_output_channels = self.move_fade_layer()
                self.output = Generator.OutputBlock(pre_output_channels)
                self.alpha = 1
                return
            
            self.init_alpha()

            if epoch_number == 0:
                print("Training input block")
                return
            
            pre_output_channels = self.move_fade_layer()

            new_layer_params = self.layer_params.pop(0)
            new_layer = Generator.SynthesisBlock(*new_layer_params)

            self.fade_in.add_fade_layer(new_layer)

            self.upsample = new_layer.upsample

            if pre_output_channels == 0:
                pre_output_channels = self.input.conv0.out_channels
            self.output = Generator.OutputBlock(pre_output_channels)


        def move_fade_layer(self):
            if len(list(self.fade_in.main.children())) == 0:
                return 0
            else:
                layer_to_move = self.fade_in.main.fade_in
                current_main_layer_count = len(list(self.main.children()))
                self.main.add_module("sb{}".format(current_main_layer_count), layer_to_move)

                pre_output_channels = layer_to_move.conv1.out_channels

                self.fade_in.clear()

                return pre_output_channels


        def add_layers(self, num_layers):
            if len(self.layer_params) == 0:
                print("All blocks are added")
                self.alpha = 1
                return

            if num_layers >= self.main_layer_count or num_layers < 0:
                count = self.main_layer_count
            else:
                count = num_layers
            
            for n in range(count):
                new_layer_params = self.layer_params.pop(0)
                layer = Generator.SynthesisBlock(*new_layer_params)
                self.main.add_module("sb{}".format(n), layer)
                print("Added block with params:{}\n".format(new_layer_params))
            
            self.output = Generator.OutputBlock(layer.conv1.out_channels)


        def forward(self, current_batch_size, tensor_map):
            """
            current_batch_size controls how many fake examples to create
            """
            self.step_alpha()

            tensor_map = self.input(current_batch_size, tensor_map)
            tensor_map = self.main(tensor_map)

            print("tensor_map: ", tensor_map.x.shape)

            tm_out = self.output(tensor_map)
            print("tm_out: ", tm_out.x.shape)
            tm_fade = self.fade_in(tensor_map)
            print("tm_fade: ", tm_fade.x.shape)

            if len(list(self.fade_in.main.children())) == 0:
                # either training just the input, or we have added all SynthesisBlocks
                return tm_out.x
            else:
                return (self.upsample(tm_out.x) * (1 - self.alpha)) + (self.alpha * tm_fade.x)


    def train(dataset, n_epochs=10):
        N_BATCHES = len(dataset)

        sg = Generator.StyleGenerator(N_BATCHES)
        mn = MappingNetwork()

        for epoch_number in range(n_epochs):
            sg.step_training_progression(epoch_number)
            print(sg)
            for batch_number, (data, label) in enumerate(dataset):

                current_batch_size = 2

                z = torch.randn(current_batch_size, 512, 1, 1)

                w = mn(z)

                tm = TensorMap(None, w)
                out = sg(current_batch_size, tm)

                out = out.mean()
                out.backward()

                sg.zero_grad()

                print(sg)
                print(10 * "-")
                print("BATCH {:4d} DONE".format(batch_number))
                print(10 * "-")
            
            graph = make_dot(out)

            graph.render(r"C:\Users\tyler.kvochick\Desktop\sg{}".format(epoch_number))

            print(10 * "-")
            print("EPOCH {:4d} DONE".format(epoch_number))
            print(10 * "-")


class Discriminator:
    """
    See https://arxiv.org/pdf/1710.10196.pdf appendix A for definition
    """
    class ConvBlock(nn.Sequential):
        
        def __init__(self, in_channels, out_channels):
            super(DiscriminatorBlock, self).__init__()
        
            layers = [
                ("conv0",      nn.Conv2d(in_channels, out_channels, (3, 3), 1, 1)),
                ("act0",       nn.LeakyReLU(negative_slope=0.2)),
                ("conv1",      nn.Conv2d(out_channels, out_channels, (3, 3), 1, 1)),
                ("act1",       nn.LeakyReLU(negative_slope=0.2)),
                ("downsample", nn.AvgPool2d((2, 2)))
            ]
            
            [self.add_module(n, l) for n, l in layers]


    class StyleDiscriminator(nn.Module):
        
        def __init__(self, layer_params=None):
            super(StyleDiscriminator, self).__init__()

            self.input = None

            self.main = nn.Sequential()
            
            if layer_params == None:
                layer_params = [
                    ( 32,  64),
                    ( 64, 128),
                    (128, 256),
                    (256, 512),
                    (512, 512),
                    (512, 512),
                    (512, 512),
                ]
            
            self.layer_params = layer_params
        
        def step_training_progression(self):
            
            current_layer_count = len(list(self.main.children()))
            
            if len(self.layer_params) == 0:
                return
            
            new_layer_params = self.layer_params.pop(0)
            
            final_out_channels = new_layer_params[1]
            
            self.main.add_module("db{}".format(current_layer_count), DiscriminatorBlock(*new_layer_params))
            print("Added block with params:{}\n".format(new_layer_params))
            
            # self.output = OutputBlock(final_out_channels)
        
        def forward(self, x):
            x = self.main(x)
            return x


if __name__ == "__main__":

    fake_dataset = [(None, None) for _ in range(10)]

    Generator.train(fake_dataset)

import torch
import torch.nn

