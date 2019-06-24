import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as nnf
from torchviz import make_dot
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision.utils as vutils
import time
import math
import functools
import operator
import copy

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
                (256, 256), #4
                (256, 256), #2
                (256, latent_dim) #1
            ]

            for n, params in enumerate(self.layer_params):
                name = "cb{}".format(n)
                layer = Encoder.ConvBlock(*(*params, name))

                self.main.add_module(name, layer)

            self.fc_mu = Encoder.D4Linear(latent_dim, latent_dim)
            self.fc_sigma = Encoder.D4Linear(latent_dim, latent_dim)

        def forward(self, container):
            
            container = self.main(container)

            x_mu = nnf.relu(self.fc_mu(container.x))
            x_sigma = nnf.relu(self.fc_sigma(container.x))

            return Container((x_mu, x_sigma), container.x_dict)
        
        def count_params(self):
            count = 0
            for p in list(self.parameters()):
                count += functools.reduce(operator.mul, list(p.shape), 1)
            return count


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
            
            if container.x_dict is not None and not self.last_layer:
                x += container.x_dict[self.block_name]

            x = self.convolution(x)
            x = self.batchnorm(x)

            if not self.last_layer:
                return Container(nnf.leaky_relu(x, negative_slope=0.1), container.x_dict)
            else:
                return Container(torch.sigmoid(x), container.x_dict)
    
    class DrawingDecoder(nn.Module):
        def __init__(self, output_channels=3, latent_dim=512):
            super().__init__()

            self.fc = Encoder.D4Linear(latent_dim, latent_dim)

            self.main = nn.Sequential()

            self.layer_params = [
                (latent_dim, 256), #2
                (256, 256), #4
                (256, 128), #8
                (128, 128), #16
                (128, 128), #32
                (128, 128), #64
                (128, 128), #128
                (128, 128), #256
                (128, output_channels) #512
            ]

            # ayer_params = [
            #     (input_channels, 8), #256
            #     (8, 16), #128
            #     (16, 32), #64
            #     (32, 64), #32
            #     (64, 128), #16
            #     (128, 256), #8
            #     (256, 512), #4
            #     (512, 512), #2
            #     (512, latent_dim) #1
            # ]

            for n, params in enumerate(self.layer_params):

                name = "cb{}".format((len(self.layer_params) - 1) - n)
                layer = Decoder.ConvBlock(*params, block_name=name)

                if n == len(self.layer_params) - 1:
                    layer.last_layer = True

                self.main.add_module(name, layer)
        

        def forward(self, container):

            x = nnf.leaky_relu(self.fc(container.x))
            container = self.main(Container(x, container.x_dict))

            return container.x
        
        def count_params(self):
            count = 0
            for p in list(self.parameters()):
                count += functools.reduce(operator.mul, list(p.shape), 1)
            return count

class Trainer:
    def kl_normal_loss(mu, var):
        return (0.5 * torch.sum(torch.exp(var) + (mu ** 2) - 1. - var, dim=(1))).mean(dim=(0)).squeeze()
    
    def resample(mu, var):
        nz, cz, hz, wz = mu.shape
        n = torch.randn_like(mu)
        return mu + torch.exp(var / 2) * n
    

    def save_images(tensor_list, path_list, nrows):
        to_image = transforms.ToPILImage()
        print("Saving images...")

        for tensor, path in zip(tensor_list, path_list):
            image_grid = vutils.make_grid(tensor, nrow=nrows, normalize=True, padding=0)
            image_grid = to_image(image_grid)

            image_grid.save(path)


    def train(downsampler, upsampler, dataloader, batch_size, n_batches, n_epochs):
        now = None
        lr = 4e-2
        best_loss = 1000
        for epoch_num in range(n_epochs):

            if epoch_num > 0 and epoch_num % 10 == 0:
                lr = lr / 1.4

            ds_optimizer = optim.SGD(downsampler.parameters(), lr=lr, momentum=0.9)
            us_optimizer = optim.SGD(upsampler.parameters(), lr=lr, momentum=0.9)

            reconstruction_criterion = nn.BCELoss(reduction="mean").cuda()

            encoding_criterion = nn.BCELoss(reduction="mean").cuda()

            for batch_num, (data, label) in enumerate(dataloader):

                ds_optimizer.zero_grad()
                us_optimizer.zero_grad()
                
                data.requires_grad = True

                data = data.cuda()

                enc = de(Container(data, None))
                z_mu, z_sd = enc.x
                z = z_sd * torch.randn_like(z_mu) + z_mu
                dec = dd(Container(z, enc.x_dict))

                mu_label = torch.zeros_like(z_mu)
                sd_label = torch.ones_like(z_sd)

                mu_loss = encoding_criterion(z_mu, mu_label)
                sd_loss = encoding_criterion(z_sd, sd_label)

                encoding_loss = (mu_loss + sd_loss)

                reconstruction_loss = reconstruction_criterion(dec, data.detach())

                total_loss = reconstruction_loss + encoding_loss

                total_loss.backward()

                # make_dot(total_loss).render(r"D:\Images\TorchViz\DrawingVAE\total-loss")

                # exit()

                ds_optimizer.step()
                us_optimizer.step()

                if total_loss.item() < best_loss:
                    best_loss = total_loss.item()
                    print(8 * "-" + " New Best Loss: {:.6f}".format(best_loss))

                    best_ds = copy.deepcopy(downsampler)
                    best_us = copy.deepcopy(upsampler)

                    now = math.floor(time.time())

                    torch.save(best_ds.cpu().state_dict(), r"D:\MLModels\DrawingVAE\{}-discriminator.pth".format(now))
                    torch.save(best_us.cpu().state_dict(), r"D:\MLModels\DrawingVAE\{}-generator.pth".format(now))


                

                if batch_num % 20 == 0:
                    update_message =\
                        "Epoch: [{:4d}/{:4d}] Batch: [{:4d}/{:4d}]\n"+\
                        "Losses: [Latent: {:.4f} Reconstruction: {:.4f} Total: {:.4f}]\n"+\
                        "Encoded Distribution: [Mean: {:.4f} StdDev: {:.4f}]\n"+\
                        "LR: {:.4f}"

                    update_message = update_message.format(
                        epoch_num,
                        n_epochs,
                        batch_num,
                        n_batches,
                        encoding_loss.item(),
                        reconstruction_loss.item(),
                        total_loss.item(),
                        z_mu.mean(),
                        z_sd.mean(),
                        lr
                        )

                    print(update_message)

                    n_rows = int(math.sqrt(batch_size))

                    tensors = [
                        data,
                        dec,
                    ]
                    tensors = [t.clone().detach().cpu() for t in tensors]

                    now = math.floor(time.time())

                    paths = [
                        r"D:\Images\DrawingVAE\{}-{}-{}-real.png".format(
                            now, epoch_num, batch_num
                        ),
                        r"D:\Images\DrawingVAE\{}-{}-{}-fake.png".format(
                            now, epoch_num, batch_num
                        ),
                    ]

                    Trainer.save_images(tensors, paths, n_rows)


    def image_dataset(path, batch_size=3):

        input_transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.Grayscale(1),
            transforms.RandomResizedCrop((512, 512)),
            # transforms.RandomVerticalFlip(),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])

        dataset = ImageFolder(path, transform=input_transform)

        dataloader = DataLoader(dataset, batch_size, num_workers=8, shuffle=True)

        return dataloader, (len(dataset.samples) // batch_size)



if __name__ == "__main__":
    data_root = r"D:\Datasets\ALotOfPlansModified"

    batch_size = 9

    dataloader, n_batches = Trainer.image_dataset(data_root, batch_size)

    model_root = r"D:\MLModels\DrawingVAE\Good"

    de = Encoder.DrawingEncoder(input_channels=1).cuda()
    dd = Decoder.DrawingDecoder(output_channels=1).cuda()

    # load_models = False
    # if load_models:
    #     models_dict = {
    #         "downsampler": torch.load(model_root + "\\1561393535-discriminator.pth"),
    #         "upsampler": torch.load(model_root + "\\1561393535-generator.pth")
    #     }

    #     de.load_state_dict(models_dict["downsampler"])
    #     dd.load_state_dict(models_dict["upsampler"])

    print("Params: [Encoder: {:,} Decoder: {:,}]".format(de.count_params(), dd.count_params()))

    Trainer.train(de, dd, dataloader, batch_size, n_batches, 500)