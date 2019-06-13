import torch
import torch.nn as nn
import torch.nn.functional as nnf

#region MappingNetwork Definitions

class MappingNetwork(nn.Sequential):
    """
    See https://arxiv.org/pdf/1812.04948.pdf section 2 for definition
    """
    def __init__(self, layer_count=8, latent_dim=512):
        super(MappingNetwork, self).__init__()

        for layer_number in range(layer_count):
            layer_name = "linear_{}".format(layer_number)
            layer = nn.Linear(latent_dim, latent_dim)
            self.add_module(layer_name, layer)

#endregion
class Generator:
    """
    See https://arxiv.org/pdf/1812.04948.pdf section 2 for definition
    """
    class A(nn.Module):
        def __init__(self, in_features, w_dim=512):
            super(Generator.A, self).__init__()
            self.affine = nn.Linear(w_dim, 2 * in_features)
        
        def forward(self, w):
            return self.affine(w).reshape(2, -1)


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
            mu_x    = x.mean(dim=(0, 2, 3)).reshape(1, -1, 1, 1)
            sigma_x = x.std(dim=(0, 2, 3)).reshape(1, -1, 1, 1)
            
            normed_x = (x - mu_x) / sigma_x
            
            y = y.reshape(2, -1, 1, 1)
            
            return (y[0, :] * x) + y[1, :]


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
        
        def forward(self, tensor_dict):

            x = tensor_dict["x"]
            w = tensor_dict["w"]
            
            x = self.upsample(x)
            x = self.conv0(x)
            x = x + self.b0()
            y = self.a0(w)
            x = self.adain0(x, y)
            
            x = self.conv1(x)
            x = x + self.b1()
            y = self.a1(w)
            x = self.adain1(x, y)
            
            return {"x": x, "w": w}


    class InputBlock(nn.Module):
        def __init__(self, in_channels, out_channels, height, width, w_dim=512):
            super(Generator.InputBlock, self).__init__()
            
            self.conv0 =    Generator.ConvBlock(in_channels, out_channels)
            self.b0 =       Generator.B(height, width, out_channels)
            self.a0 =       Generator.A(out_channels, w_dim=w_dim)
            self.adain0 =   Generator.AdaIN()
            
            self.b1 =       Generator.B(height, width, out_channels)
            self.a1 =       Generator.A(out_channels, w_dim=w_dim)
            self.adain1 =   Generator.AdaIN()
        
        def forward(self, tensor_dict):

            x = tensor_dict["x"]
            w = tensor_dict["w"]
            
            x = self.conv0(x)
            x = x + self.b0()
            y = self.a0(w)
            x = self.adain0(x, y)

            x = x + self.b1()
            y = self.a1(w)
            x = self.adain1(x, y)
            
            return {"x": x, "w": w}


    class OutputBlock(nn.Module):
        def __init__(self, input_channels):
            super(Generator.OutputBlock, self).__init__()
            
            self.to_rgb = nn.Conv2d(input_channels, 3, (1, 1))
        
        def forward(self, tensor_dict):
            x = tensor_dict["x"]
            w = tensor_dict["w"]
            
            x = self.to_rgb(x)
            
            return {"x": x, "w": w}


    class FadeIn(nn.Module):
        def __init__(self):
            super(Generator.FadeIn, self).__init__()

            self.alpha = 0


    class StyleGenerator(nn.Module):
        def __init__(self, input_layer=None, layer_params=None, w_dim=512):
            super(Generator.StyleGenerator, self).__init__()
            
            if input_layer == None:
                input_layer = Generator.InputBlock(512, 512, 4, 4, w_dim=w_dim)
            
            self.input = input_layer

            self.main = nn.Sequential()

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

        def step_training_progression(self):
            
            current_layer_count = len(list(self.main.children()))
            
            if len(self.layer_params) == 0:
                return
            
            new_layer_params = self.layer_params.pop(0)
            
            final_out_channels = new_layer_params[1]
            
            self.main.add_module("sb{}".format(current_layer_count), SynthesisBlock(*new_layer_params))
            print("Added block with params:{}\n".format(new_layer_params))
            
            self.output = OutputBlock(final_out_channels)

        def forward(self, tensor_dict):
            tensor_dict = self.input(tensor_dict)
            tensor_dict = self.main(tensor_dict)

            return self.output(tensor_dict)

#region Discriminator Definitions

class DiscriminatorBlock(nn.Sequential):
    
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

#endregion

#region Testing
if __name__ == "__main__":
    sg = Generator.StyleGenerator()

    print(sg)
#endregion