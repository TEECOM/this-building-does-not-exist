import torch
import torch.nn as nn
import torch.nn.functional as nnf
from torch.utils.data import DataLoader
import functools, operator
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
import torchvision.utils as vutils
import math
import time
from multiprocessing import Process
from torchviz import make_dot


class MappingNetwork(nn.Module):
    def __init__(self, latent_dim=512, n_layers=8):
        super(MappingNetwork, self).__init__()

        self.main = nn.Sequential()

        for n in range(n_layers):
            name = "fc-{}".format(n)
            layer = nn.Linear(latent_dim, latent_dim)
            self.main.add_module(name, layer)

            # name = "act-{}".format(n)
            # layer = nn.LeakyReLU(latent_dim, latent_dim)
            # self.main.add_module(name, layer)
    
    def forward(self, z):
        nz, lz, hz, wz = z.shape
        return nnf.leaky_relu(self.main(z.view(nz, lz)))


class Container:
    def __init__(self, x, w):
        self.x = x
        self.w = w


class Generator:

    class A(nn.Module):
        def __init__(self, in_channels, latent_dim=512):
            super(Generator.A, self).__init__()

            self.in_channels = in_channels

            self.style_s = nn.Linear(latent_dim, in_channels)
            self.style_b = nn.Linear(latent_dim, in_channels)

        def forward(self, w):
            nw, cw = w.shape

            y_s = nnf.leaky_relu(self.style_s(w)).view(nw, self.in_channels, 1, 1)
            y_b = nnf.leaky_relu(self.style_b(w)).view(nw, self.in_channels, 1, 1)

            return (y_s, y_b)
    

    class AdaIN(nn.Module):
        def __init__(self):
            super(Generator.AdaIN, self).__init__()
            
        def forward(self, x, y):
            # y is coming from an A block
            # x is a conv output
            n, c, h, w = x.shape
            mu_x    = x.mean(dim=(0, 2, 3), keepdim=True)
            sigma_x = x.std(dim=(0, 2, 3), keepdim=True)
            
            normed_x = (x - mu_x) / sigma_x
            
            y_s, y_b = y
            
            return (y_s * x) + y_b



    class ConvBlock(nn.Module):
        def __init__(self, in_channels, out_channels, latent_dim=512):
            super(Generator.ConvBlock, self).__init__()

            self.latent_dim = latent_dim

            self.a = Generator.A(in_channels, latent_dim=latent_dim)

            self.ada_in = Generator.AdaIN()

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
        
        def forward(self, container):
            nx, cx, hx, wx = container.x.shape

            y = self.a(container.w)

            x = self.ada_in(container.x, y)

            x = self.conv(container.x)
            # y = self.style0(container.w.view(nx, self.latent_dim))

            # x = self.bn(x * y.view(nx, self.conv.out_channels, 1, 1))

            x = self.bn(x)

            # xm = container.x.mean(dim=(0, 1), keepdim=True)
            # xm = nnf.interpolate(xm, size=(2 * hx, 2 * wx), mode="bilinear", align_corners=True)

            x = self.act(x)

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
                (128, 128), #64
                (128, 128), #128
                (128, 128), #256
                (128, 128), #512
                (128, output_channels), #1024
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
                const = self.constant.expand(nz, -1, -1, -1)

                # const = nnf.leaky_relu(const * w.view(nz, -1, 1, 1))

                out = self.main(Container(const, w))
                return out.x
            if mode == "ae":

                # x = nnf.leaky_relu(x * w.view(nz, -1, 1, 1))

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

            # x = x.mean(dim=(0, 1), keepdim=True)
            # x = nnf.interpolate(x, size=(hx // 2, wx // 2), mode="bilinear", align_corners=True)

            y = self.bn(y)

            return self.act(y)
    
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
    def save_images(tensor_list, path_list, nrows):
        to_image = transforms.ToPILImage()
        print("Saving images...")

        for tensor, path in zip(tensor_list, path_list):
            image_grid = vutils.make_grid(tensor, nrow=nrows, normalize=True, padding=0)
            image_grid = to_image(image_grid)

            image_grid.save(path)

    def train(n_epochs, n_batches, dataloader, cuda_idx, models_dict=None):
        to_image = transforms.ToPILImage()

        dd = Discriminator.DrawingDiscriminator(input_channels=1).cuda(cuda_idx)
        dg = Generator.DrawingGenerator(output_channels=1).cuda(cuda_idx)

        print("DD Params: ", dd.count_params())
        print("DG Params: ", dg.count_params())

        if models_dict is not None:
            dd.load_state_dict(models_dict["discriminator"])
            dg.load_state_dict(models_dict["generator"])

        ae_criterion = nn.BCELoss(reduction="mean").cuda(cuda_idx)
        discriminator_criterion = nn.BCELoss(reduction="mean").cuda(cuda_idx)
        generator_criterion = nn.BCELoss(reduction="mean").cuda(cuda_idx)

        for epoch_number in range(n_epochs):

            lr = 5e-3

            if epoch_number % 10 == 0 and epoch_number != 0:
                lr = lr / 1.1

            dd_optimizer = torch.optim.Adam(dd.parameters(), lr=lr)
            dg_optimizer = torch.optim.Adam(dg.parameters(), lr=lr)

            for batch_number, (data, label) in enumerate(dataloader):

                batch_size, channels, height, width = data.shape

                data = data.cuda(cuda_idx)
                data.requires_grad = True

                real_label = torch.zeros(batch_size, 2).cuda(cuda_idx)
                real_label[:, 0] = 1.0

                fake_label = torch.zeros(batch_size, 2).cuda(cuda_idx)
                fake_label[:, 1] = 1.0

                # Autoencoder

                z = torch.randn(batch_size, 512, 1, 1).cuda(cuda_idx)

                # encoded = dd(data, mode="ae")
                # decoded = dg(z, encoded, mode="ae")

                # reconstruction_loss = ae_criterion(decoded, data.detach())

                # reconstruction_loss.backward()

                # dd_optimizer.step()
                # dg_optimizer.step()

                # dd_optimizer.zero_grad()
                # dg_optimizer.zero_grad()

                # Generate fake images

                fake_images = dg(z, None, mode="generator")

                # Train on real

                real_classification = dd(data, mode="discriminator")
                real_loss = discriminator_criterion(real_classification, real_label)

                # real_loss = torch.tensor(0)

                real_loss.backward()

                # Train on fake

                fake_classification = dd(fake_images.detach(), mode="discriminator")
                fake_loss = discriminator_criterion(fake_classification, fake_label)

                # fake_loss = torch.tensor(0)

                fake_loss.backward()

                dd_optimizer.step()
                dd_optimizer.zero_grad()

                # Train generator on how well it fooled discriminator

                gen_classification = dd(fake_images, mode="discriminator")
                gen_loss = discriminator_criterion(gen_classification, real_label)

                # gen_loss = torch.tensor(0)

                gen_loss.backward()

                dg_optimizer.step()

                dg_optimizer.zero_grad()

                # Progress tracking

                # if epoch_number == 0 and batch_number == 0:
                #     make_dot(decoded).render(r"D:\Images\TorchViz\SimpleStyleGan\decoded")
                #     make_dot(real_loss).render(r"D:\Images\TorchViz\SimpleStyleGan\real")
                #     make_dot(gen_loss).render(r"D:\Images\TorchViz\SimpleStyleGan\gen")
                
                if batch_number % 20 == 0:
                    update_message =\
                        "Epoch: [{:4d}/{:4d}] Batch: [{:4d}/{:4d}]\n"+\
                        "Losses: [Real: {:.4f} Fake: {:.4f} Generator {:.4f}]\n"

                    update_message = update_message.format(
                        epoch_number + 1,
                        n_epochs,
                        batch_number,
                        n_batches,
                        real_loss.item(),
                        fake_loss.item(),
                        # reconstruction_loss.item(),
                        gen_loss.item(),
                        )

                    print(update_message)

                    n_rows = int(math.sqrt(batch_size))

                    tensors = [
                        fake_images,
                        data,
                        # decoded
                    ]
                    tensors = [t.clone().detach().cpu() for t in tensors]

                    paths = [
                        r"D:\Images\SimpleStyleGAN\{}-{}-{}-fake.png".format(
                            math.floor(time.time()), epoch_number, batch_number
                        ),
                        r"D:\Images\SimpleStyleGAN\{}-{}-{}-real.png".format(
                            math.floor(time.time()), epoch_number, batch_number
                        ),
                        # r"D:\Images\SimpleStyleGAN\{}-{}-{}-decoded.png".format(
                        #     math.floor(time.time()), epoch_number, batch_number
                        # )
                    ]

                    DrawingGAN.save_images(tensors, paths, n_rows)

            # Per epoch
            torch.save(dd.state_dict(), r"D:\MLModels\SimpleStyleGAN\{}-discriminator.pth".format(math.floor(time.time())))
            torch.save(dg.state_dict(), r"D:\MLModels\SimpleStyleGAN\{}-generator.pth".format(math.floor(time.time())))
    

    def image_dataset(path, batch_size=3):

        input_transform = transforms.Compose([
            transforms.Resize((1024, 1024)),
            transforms.Grayscale(1),
            transforms.ToTensor()
        ])

        dataset = ImageFolder(path, transform=input_transform)

        dataloader = DataLoader(dataset, batch_size, num_workers=8, shuffle=True)

        return dataloader, (len(dataset.samples) // batch_size)


                

if __name__ == "__main__":

    data_root = r"D:\Datasets\ALotOfPlansModified"

    model_root = r"D:\MLModels\SimpleStyleGAN"

    # models_dict = {
    #     "discriminator": torch.load(model_root + r"\_1560939083-discriminator.pth"),
    #     "generator": torch.load(model_root + r"\_1560939083-generator.pth")
    # }

    dataloader, n_batches = DrawingGAN.image_dataset(data_root, batch_size=4)

    DrawingGAN.train(1000, n_batches, dataloader, 1, models_dict=None)





    
    