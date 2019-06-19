import torch
import torch.nn as nn
import torch.nn.functional as nnf
import functools, operator


class MappingNetwork(nn.Module):
    def __init__(self, latent_dim=512, n_layers=8):
        super(MappingNetwork, self).__init__()

        self.main = nn.Sequential()

        for n in range(n_layers):
            name = "fc-{}".format(n)
            layer = nn.Linear(latent_dim, latent_dim)
            self.main.add_module(name, layer)
    
    def forward(self, z):
        nz, lz, hz, wz = z.shape
        return self.main(z.view(nz, lz))


class Container:
    def __init__(self, x, w):
        self.x = x
        self.w = w


class Generator:
    class ConvBlock(nn.Module):
        def __init__(self, in_channels, out_channels, latent_dim=512):
            super(Generator.ConvBlock, self).__init__()

            self.latent_dim = latent_dim

            self.conv = nn.ConvTranspose2d(
                in_channels,
                out_channels,
                4,
                2,
                1,
                bias=False
                )

            self.bn = nn.BatchNorm2d(out_channels)
            self.act = nn.LeakyReLU()
            self.style0 = nn.Linear(latent_dim, out_channels)
        
        def forward(self, container):
            nx, cx, hx, wx = container.x.shape

            x = self.conv(container.x)
            y = self.style0(container.w.view(nx, self.latent_dim))

            x = self.bn(x * y.view(nx, self.conv.out_channels, 1, 1))

            xm = container.x.mean(dim=(0, 1), keepdim=True)
            xm = nnf.interpolate(xm, size=(2 * hx, 2 * wx), mode="bilinear", align_corners=True)

            x = self.act(x + xm)

            return Container(x, container.w)
    
    class DrawingGenerator(nn.Module):
        def __init__(self, latent_dim=512, output_channels=3):
            super(Generator.DrawingGenerator, self).__init__()

            self.latent_dim = latent_dim

            self.mapping_network = MappingNetwork(latent_dim=latent_dim)

            self.constant = nn.Parameter(
                data=torch.randn(1, latent_dim, 2, 2),
                requires_grad=True
                )

            self.main = nn.Sequential()

            self.layer_params = [
                (512, 256), #4
                (256, 128), #8
                (128, 128), #16
                (128, 128), #32
                (128,  64), #64
                ( 64,  32), #128
                ( 32,  16), #256
                ( 16,   8), #512
                (  8,   output_channels), #1024
            ]

            for n, params in enumerate(self.layer_params):
                name = "cb-{}".format(n)
                layer = Generator.ConvBlock(*params)

                last_layer = n == len(self.layer_params) - 1

                if last_layer:
                    layer.act = nn.Sigmoid()
                
                self.main.add_module(name, layer)
        

        def count_params(self):
            count = 0
            for p in list(self.parameters()):
                count += functools.reduce(operator.mul, list(p.shape), 1)
            return count

            
        def forward(self, z, x, mode="generator"):
            nz, cz, hz, wz = z.shape
            w = self.mapping_network(z)

            if mode == "generator":
                out = self.main(Container(self.constant.expand(nz, -1, -1, -1), w))
                return out.x
            if mode == "ae":
                out = self.main(Container(x, w))
                return out.x


class Discriminator:
    class ConvBlock(nn.Module):
        def __init__(self, in_channels, out_channels, kernel=4, stride=2, pad=1):
            super(Discriminator.ConvBlock, self).__init__()

            self.conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel,
                stride,
                pad,
                bias=False
                )
            
            self.bn = nn.BatchNorm2d(out_channels)

            self.act = nn.LeakyReLU()
        
        def forward(self, x):
            nx, cx, hx, wx = x.shape

            y = self.conv(x)

            x = x.mean(dim=(0, 1), keepdim=True)
            x = nnf.interpolate(x, size=(hx // 2, wx // 2), mode="bilinear", align_corners=True)

            y = self.bn(y)

            return self.act(y + x)
    
    class DrawingDiscriminator(nn.Module):
        def __init__(self, input_channels=3, latent_dim=2048):
            super(Discriminator.DrawingDiscriminator, self).__init__()

            self.latent_dim = latent_dim

            self.main = nn.Sequential()

            self.layer_params = [
                (  input_channels,   8),
                (  8,  16),
                ( 16,  32),
                ( 32,  64),
                ( 64, 128),
                (128, 128),
                (128, 128),
                (128, 256),
                (256, 512),
                (512, latent_dim, 2, 1, 0)
            ]

            for n, params in enumerate(self.layer_params):
                name = "cb-{}".format(n)
                layer = Discriminator.ConvBlock(*params)

                self.main.add_module(name, layer)
            
            self.fc = nn.Linear(latent_dim, 2)
            self.lrelu = nn.LeakyReLU()
            self.sig = nn.Sigmoid()
        

        def count_params(self):
            count = 0
            for p in list(self.parameters()):
                count += functools.reduce(operator.mul, list(p.shape), 1)
            return count


        def forward(self, x, mode="discriminator"):
            nx, cx, hx, wx = x.shape
            y = self.main(x)

            if mode == "discriminator":
                y = self.fc(y.view(nx, self.latent_dim))
                return self.sig(y)
            if mode == "ae":
                wh = self.latent_dim // 512
                return self.lrelu(y.reshape(nx, 512, wh//2, wh//2))


class DrawingGAN:
    def train(n_epochs, dataloader, cuda_idx):

        dd = Discriminator.DrawingDiscriminator().cuda(cuda_idx)
        dg = Generator.DrawingGenerator().cuda(cuda_idx)

        ae_criterion = nn.BCELoss(reduction="mean").cuda(cuda_idx)
        discriminator_criterion = nn.BCELoss(reduction="mean").cuda(cuda_idx)
        generator_criterion = nn.BCELoss(reduction="mean").cuda(cuda_idx)

        for epoch in range(n_epochs):

            dd_optimizer = torch.optim.Adam(dd.parameters(), lr=1e-3)
            dg_optimizer = torch.optim.Adam(dg.parameters(), lr=1e-3)

            for batch_num, (data, label) in enumerate(dataloader):
                batch_size, channels, height, width = data.shape

                data = data.cuda(cuda_idx)
                data.requires_grad = True

                real_label = torch.zeros(batch_size, 2).cuda(cuda_idx)
                real_label[:, 0] = 1.0

                fake_label = torch.zeros(batch_size, 2).cuda(cuda_idx)
                fake_label[:, 1] = 1.0

                # Autoencoder

                z = torch.randn(batch_size, 512, 1, 1).cuda(cuda_idx)

                encoded = dd(data, mode="ae")
                decoded = dg(z, encoded, mode="ae")

                reconstruction_loss = ae_criterion(decoded, data)

                reconstruction_loss.backward()

                dd_optimizer.step()
                dg_optimizer.step()

                dd_optimizer.zero_grad()
                dg_optimizer.zero_grad()

                # Generate fake images

                fake_images = dg(z, None, mode="generator")

                # Train on real

                real_classification = dd(data, mode="discriminator")
                real_loss = discriminator_criterion(real_classification, real_label)

                real_loss.backward()

                # Train on fake

                fake_classification = dd(fake_images.detach(), mode="discriminator")
                fake_loss = discriminator_criterion(fake_classification, fake_label)

                fake_loss.backward()

                dd_optimizer.step()
                dd_optimizer.zero_grad()

                # Train generator on how well it fooled discriminator

                gen_classification = dd(fake_images, mode="discriminator")
                gen_loss = discriminator_criterion(gen_classification, real_label)

                gen_loss.backward()

                dg_optimizer.step()

                dg_optimizer.zero_grad()


                

if __name__ == "__main__":
    from PIL import Image
    import matplotlib.pyplot as plt
    from torchviz import make_dot

    dd = Discriminator.DrawingDiscriminator(input_channels=1).cuda()
    dg = Generator.DrawingGenerator(output_channels=1).cuda()

    z = torch.randn(4, 512, 1, 1).cuda()

    images = dg(z, None, mode="generator")
    print(images.shape)
    classification = dd(images, mode="discriminator")
    print(classification.shape)

    # make_dot(classification).render(
    #     r"C:\Users\tyler.kvochick\Desktop\simple-style-gan-d"
    #     )
    

    encoding = dd(images, mode="ae")
    print(encoding.shape)
    decoding = dg(z, encoding, mode="ae")
    print(decoding.shape)

    # make_dot(decoding).render(
    #     r"C:\Users\tyler.kvochick\Desktop\simple-style-gan-a"
    #     )

    print(dd.count_params())
    print(dg.count_params())
    input("waiting")





    
    
