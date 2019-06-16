import torch
import torch.nn as nn
import torch.nn.functional as nnf
from torchviz import make_dot
import matplotlib.pyplot as plt
from collections import OrderedDict as odict


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


class Settings:
    BatchSizeProgression = [128, 128, 128, 64, 32, 16, 8, 4] # For 12GB GPU


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
            
            self.noise_image = torch.nn.Parameter(data=torch.randn(1, 1, height, width), requires_grad=False)
            
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
            self.act = nn.LeakyReLU(0.2)
        
        def forward(self, tensor_map):
            x = tensor_map.x
            w = tensor_map.w
            
            x = self.to_rgb(x)
            x = self.act(x)
            
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
            self.main.add_module(
                "output",
                Generator.OutputBlock(layer.conv1.out_channels)
                )
        
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

            self.mapping_network = MappingNetwork(latent_dim=w_dim)
            
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
            
            self.alpha_progression = list(torch.arange(0, 1 + self.alpha_step, self.alpha_step, dtype=torch.float32).flip(0))
            
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

                if pre_output_channels != 0:
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


        def forward(self, current_batch_size, z):
            """
            current_batch_size controls how many fake examples to create
            """
            self.step_alpha()

            w = self.mapping_network(z)

            tensor_map = TensorMap(None, w)

            tensor_map = self.input(current_batch_size, tensor_map)
            tensor_map = self.main(tensor_map)

            tm_out = self.output(tensor_map)

            tm_fade = self.fade_in(tensor_map)

            if len(list(self.fade_in.main.children())) == 0:
                # either training just the input, or we have added all SynthesisBlocks
                return tm_out.x
            else:
                return (self.upsample(tm_out.x) * (1.0 - self.alpha))\
                    + (self.alpha * tm_fade.x)


    def train(dataset, n_epochs=11):
        N_BATCHES = len(dataset)

        sg = Generator.StyleGenerator(N_BATCHES)

        for epoch_number in range(n_epochs):
            sg.step_training_progression(epoch_number)
            sg.cuda()

            for batch_number, (data, label) in enumerate(dataset):
                current_batch_size = 4

                z = torch.randn(current_batch_size, 512, 1, 1).cuda()

                out = sg(current_batch_size, z)

                print("Alpha: {}".format(sg.alpha))

                out = out.mean()
                out.backward()

                sg.zero_grad()

                #print(sg)
                print(10 * "-")
                print("BATCH {:4d} DONE".format(batch_number))
                print(10 * "-")
            
            #graph = make_dot(out)

            #graph.render(r"C:\Users\tyler.kvochick\Desktop\sg{}".format(epoch_number))

            print(10 * "-")
            print("EPOCH {:4d} DONE".format(epoch_number))
            print(10 * "-")
    
    def get_max_batch_size(cuda=True):
        sg = Generator.StyleGenerator(1)

        sg.add_layers(1)

        batch_size = 1

        while True:
            try:
                z = torch.randn(batch_size, 512, 1, 1, requires_grad=True)
                if cuda:
                    sg.cuda()
                    z = z.cuda()
                
                out = sg(batch_size, z)

                out.mean().backward()

                sg.zero_grad()

                batch_size += 1
            
            except Exception as e:
                print(e)
                break


class Discriminator:
    """
    See https://arxiv.org/pdf/1710.10196.pdf appendix A for definition
    """
    class ConvBlock(nn.Sequential):
        
        def __init__(self, in_channels, out_channels):
            super(Discriminator.ConvBlock, self).__init__()

            self.in_channels = in_channels
            self.out_channels = out_channels
        
            layers = [
                ("conv0",      nn.Conv2d(in_channels, out_channels, (3, 3), 1, 1)),
                ("act0",       nn.LeakyReLU(negative_slope=0.2)),
                ("conv1",      nn.Conv2d(out_channels, out_channels, (3, 3), 1, 1)),
                ("act1",       nn.LeakyReLU(negative_slope=0.2)),
                ("downsample", nn.AvgPool2d((2, 2)))
            ]
            
            [self.add_module(n, l) for n, l in layers]
    

    class OutputBlock(nn.Module):
        def __init__(self, num_features=512):
            super(Discriminator.OutputBlock, self).__init__()

            self.in_channels = num_features

            self.conv0 = nn.Conv2d(num_features, num_features, (3, 3), 1, 1)
            self.act0 = nn.LeakyReLU(0.2)
            self.conv1 = nn.Conv2d(num_features, num_features, (4, 4))
            self.act1 = nn.LeakyReLU(0.2)
            self.fc = nn.Linear(num_features, 2)
        
        def forward(self, x):

            x = self.conv0(x)
            x = self.act0(x)
            x = self.conv1(x)
            x = self.act1(x)
            n, c, h, w = x.shape
            x = self.fc(x.view(n, c * h * w))

            return x


    class FromRGB(nn.Conv2d):
        def __init__(self, out_channels):
            super(Discriminator.FromRGB, self).__init__(3, out_channels, (1, 1))


    class FadeIn(nn.Module):
        def __init__(self):
            super(Discriminator.FadeIn, self).__init__()

            self.from_rgb = None
            self.main = nn.Sequential()
            self.current_layer_name = ""
        
        @property
        def has_fade_layer(self):
            main_layer_count = len(list(self.main.children()))
            return main_layer_count > 0
        
        def add_fade_layer(self, name, layer):
            if self.has_fade_layer:
                self.clear()
            self.from_rgb = Discriminator.FromRGB(layer.in_channels)
            self.current_layer_name = name
            self.main.add_module("fade_in", layer)
        
        def clear(self):
            del(self.main.fade_in)
            del(self.from_rgb)
            self.current_layer_name = ""
        
        def forward(self, x):
            x = self.from_rgb(x)
            x = self.main.fade_in(x)
            return x


    class StyleDiscriminator(nn.Module):
        
        def __init__(self, n_batches, layer_params=None):
            super(Discriminator.StyleDiscriminator, self).__init__()
            
            self.n_batches = n_batches

            self.init_alpha()

            self.from_rgb = None

            self.input = None

            self.main = nn.Sequential()

            self.main.add_module("output", Discriminator.OutputBlock())

            self.fade_in = Discriminator.FadeIn()
            
            if layer_params == None:
                layer_params = [
                    ( 16,  32),
                    ( 32,  64),
                    ( 64, 128),
                    (128, 256),
                    (256, 512),
                    (512, 512),
                    (512, 512),
                    (512, 512),
                ]

            self.layer_params = layer_params

            self.conv_blocks = [
                ("cb{}".format(n), Discriminator.ConvBlock(*params))
                    for n, params in enumerate(layer_params)
                ]

            self.downsample = None

            self.set_from_rgb_layer()
        

        @property
        def main_layer_count(self):
            return len(list(self.main.children()))
        

        @property
        def post_rgb_channels(self):
            main_children = list(self.main.children())
            return main_children[0].in_channels
        

        def set_from_rgb_layer(self):
            correct_rgb_out = None
            if self.from_rgb is not None:
                correct_rgb_out =\
                    self.from_rgb.out_channels == self.post_rgb_channels
            
            if not correct_rgb_out:
                # a smarter way would be to just concat enough channels
                # to the already learned tensor?
                self.from_rgb = Discriminator.FromRGB(self.post_rgb_channels)


        def init_alpha(self):
            self.alpha_step = (1 / self.n_batches)
            
            self.alpha_progression = list(
                torch.arange(
                    0,
                    1 + self.alpha_step,
                    self.alpha_step,
                    dtype=torch.float32).flip(0)
                    )
            
            self.alpha = 0
        

        def step_alpha(self):
            if len(self.alpha_progression) == 0:
                return
            self.alpha = self.alpha_progression.pop(0)
            
        
        def step_training_progression(self, epoch_number):
            if len(self.conv_blocks) == 0:
                print("All blocks added")
                self.move_fade_layer()
                self.fade_in.clear()
                self.alpha = 1.0
                return
            
            if epoch_number == 0:
                print("Training output layer.")
                return
            
            name, new_layer = self.conv_blocks.pop()

            self.downsample = new_layer.downsample

            self.move_fade_layer()

            self.fade_in.add_fade_layer(name, new_layer)

            self.init_alpha()
        

        def move_fade_layer(self):

            if not self.fade_in.has_fade_layer:
                return
            
            newly_faded_layer = self.fade_in.main.fade_in

            name_layer_pairs = odict([
                (self.fade_in.current_layer_name, newly_faded_layer)
            ])
            
            name_layer_pairs.update(self.main._modules)

            del(self.main)
            self.main = nn.Sequential(odict(name_layer_pairs))
            self.set_from_rgb_layer()
            

        
        def add_layers(self, num_layers):
            if num_layers > len(self.conv_blocks):
                num_layers = len(self.conv_blocks)
            for n in range(num_layers):
                name, layer = self.conv_blocks.pop()
                self.main.add_module(name, layer)
            self.set_from_rgb_layer()
        
            
        def forward(self, x):

            if self.fade_in.has_fade_layer:
                self.step_alpha()
                xf = self.fade_in(x)
                xm = self.downsample(x)
                xm = self.main(self.from_rgb(xm))
                xf = self.main(xf)
                return ((1.0 - self.alpha) * xm) + (xf * self.alpha)

            return self.main(self.from_rgb(x))

    def train(dataset, n_epochs=2):
        N_BATCHES = len(dataset)

        sd = Discriminator.StyleDiscriminator(N_BATCHES)
        sg = Generator.StyleGenerator(N_BATCHES)

        for epoch_number in range(n_epochs):
            sd.step_training_progression(epoch_number)
            sg.step_training_progression(epoch_number)
            print(sd)
            sd.cuda()
            print(sg)
            sg.cuda()
            for batch_number, (data, label) in enumerate(dataset):

                z = torch.randn(1, 512, 1, 1, requires_grad=True).cuda()

                out = sg.forward(1, z)
                print("OUT SHAPE {}".format(out.shape))
                out = sd(out)

                out = out.mean()

                sg.zero_grad()
                sd.zero_grad()

                if epoch_number == 8 and batch_number == 0:
                    graph = make_dot(out)
                    graph.render(r"C:\Users\tyler.kvochick\Desktop\sg")


 
                print(10 * "-")
                print("BATCH {:4d} DONE".format(batch_number))
                print(10 * "-")
            
            # graph = make_dot(out)
            # graph.render(r"C:\Users\tyler.kvochick\Desktop\sg{}".format(epoch_number))
            print(10 * "-")
            print("EPOCH {:4d} DONE".format(epoch_number))
            print(10 * "-")

if __name__ == "__main__":

    fake_dataset = [(None, None) for _ in range(10)]

    Discriminator.train(fake_dataset, n_epochs=9)


