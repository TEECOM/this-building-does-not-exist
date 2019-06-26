import torch
import torch.nn as nn
import torch.nn.functional as nnf
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import math
import time
import functools
import operator
import copy

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
                (in_channels, fc * 8),
                (fc * 8, fc * 8),
                (fc * 8, fc * 8),
                (fc * 8, fc * 8),
                (fc * 8, fc * 16),
                (fc * 16, fc * 32),
                (fc * 32, fc * 64),
                (fc * 64, fc * 64),
                (fc * 64, encoding_dim, 2, 2, 0)
            ]

            for num, params in enumerate(self.layer_params):
                name = "cb{}".format(num)
                layer = conv_block_class(*params)

                self.main.add_module(name, layer)
            
            self.encoding = Linear4D(encoding_dim, encoding_dim)
            self.classifier = Linear4D(encoding_dim, out_categories)
            
            self.activation = lambda x: nnf.leaky_relu(x, negative_slope=0.2)
        
        def forward(self, x, mode="discriminator"):
            y = self.main(x)
            y = self.activation(self.encoding(y))
            
            if mode == "ae":
                return y
            
            if mode == "discriminator":
                y = torch.sigmoid(self.classifier(y))
                return y
        
        def count_parameters(self):
            params = 0
            for p in self.parameters():
                params += functools.reduce(operator.mul, list(p.shape), 1)
            return params

class Generator:
    class ConvBlock(nn.Module):
        def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1):
            super().__init__()

            self.conv = nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False
                )
            
            self.bn = nn.BatchNorm2d(out_channels)
        
        def forward(self, x):
            y = self.conv(x)
            y = self.bn(y)

            return nnf.leaky_relu(y, negative_slope=0.2)
    
    class Generator(nn.Module):
        def __init__(self, conv_block_class, latent_dim=128, feature_coeff=8, out_channels=3):
            super().__init__()

            self.linear = Linear4D(latent_dim, latent_dim)

            self.main = nn.Sequential()

            fc = feature_coeff

            self.layer_params = [
                (latent_dim, fc * 64),
                (fc * 64, fc * 64),
                (fc * 64, fc * 32),
                (fc * 32, fc * 32),
                (fc * 32, fc * 32),
                (fc * 32, fc * 32),
                (fc * 32, fc * 32),
                (fc * 32, fc * 32),
                (fc * 32, out_channels)
            ]

            for num, p in enumerate(self.layer_params):
                name = "cb{}".format(num)
                layer = conv_block_class(*p)

                self.main.add_module(name, layer)
            
        def forward(self, x, mode="generator", mu=0, std=1):

            if mode == "generator":
                x = std * x + mu

            y = nnf.leaky_relu(self.linear(x), negative_slope=0.2)
            y = self.main(y)
            return torch.sigmoid(y)
        
        def count_parameters(self):
            params = 0
            for p in self.parameters():
                params += functools.reduce(operator.mul, list(p.shape), 1)
            return params


class AEGeAN:
    def save_images(tensor_path_list, nrows):
        to_image = transforms.ToPILImage()
        print("Saving images...")

        for tensor, path in tensor_path_list:
            image_grid = vutils.make_grid(tensor.clone().detach().cpu(), nrow=nrows, normalize=True, padding=0)
            image_grid = to_image(image_grid)

            image_grid.save(path)
    
    def image_dataset(path, batch_size=4):

        input_transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.Grayscale(1),
            transforms.ToTensor(),
        ])

        dataset = ImageFolder(path, transform=input_transform)

        dataloader = DataLoader(dataset, batch_size, num_workers=8, shuffle=True)

        return dataloader, (len(dataset.samples) // batch_size)
    
    def train(dataloader, n_epochs, n_batches, experiment_root, models=None):

        G = Generator.Generator(Generator.ConvBlock, feature_coeff=4, out_channels=1)
        D = Discriminator.Discriminator(Discriminator.ConvBlock, in_channels=1, feature_coeff=4)

        print("G params: {:,}".format(G.count_parameters()))
        print("D params: {:,}".format(D.count_parameters()))

        if models is not None:
            G.load_state_dict(models["generator"])
            D.load_state_dict(models["discriminator"])

        G.cuda()
        D.cuda()

        ae_criterion = nn.BCELoss(reduction="mean").cuda()
        d_criterion = nn.BCELoss(reduction="mean").cuda()

        for epoch_num in range(n_epochs):

            lr = 5e-3

            g_optim = optim.Adam(G.parameters(), lr=lr)
            d_optim = optim.SGD(D.parameters(), lr=lr, momentum=0.9)

            for batch_num, (data, label) in enumerate(dataloader):

                batch_size, channels, height, width = data.shape

                real_label = torch.zeros(batch_size, 2, 1, 1).cuda()
                real_label[:, 0, :, :] = 1.0
                fake_label = torch.zeros(batch_size, 2, 1, 1).cuda()
                fake_label[:, 1, :, :] = 1.0

                data = data.cuda()
                # data.sub_(0.5).mul_(2)

                data.requires_grad = True

                encoded = D(data, mode="ae")
                e_mu = encoded.mean().detach()
                e_std = encoded.std().detach()
                decoded = G(encoded, mode="ae")

                # reconstruction_loss = ae_criterion(decoded, data.detach().div(2).add(.5))
                reconstruction_loss = ae_criterion(decoded, data.detach())

                reconstruction_loss.backward()

                g_optim.step()
                d_optim.step()

                g_optim.zero_grad()
                d_optim.zero_grad()

                # GAN

                c_real = D(data, mode="discriminator")

                real_loss = d_criterion(c_real, real_label)

                z = torch.randn(batch_size, 128, 1, 1).cuda()
                z.requires_grad = True

                fake_images = G(z, mode="generator", mu=e_mu, std=e_std)

                c_fake = D(fake_images.detach(), mode="discriminator")

                fake_loss = d_criterion(c_fake, fake_label)

                d_total_loss = real_loss + fake_loss
                d_total_loss.backward()

                d_optim.step()
                d_optim.zero_grad()

                c_fooled = D(fake_images, mode="discriminator")

                fooled_loss = d_criterion(c_fooled, real_label)

                fooled_loss.backward()

                g_optim.step()
                g_optim.zero_grad()
                now = math.floor(time.time())

                if batch_num % 20 == 0:

                    msg =\
                        "Epoch [{:4d}/{:4d}] Batch [{:4d}/{:4d}]\n"+\
                        "Encoded: [Mu: {:.4f} Sd: {:.4f}]\n"+\
                        "Losses [AE: {:.4f} Real: {:.4f} Fake: {:.4f} Generator: {:.4f}]"
                        # "Losses [AE: {:.4f}]\n"+\
                    
                    msg = msg.format(
                        epoch_num,
                        n_epochs,
                        batch_num,
                        n_batches,
                        e_mu.item(),
                        e_std.item(),
                        reconstruction_loss.item(),
                        real_loss.item(),
                        fake_loss.item(),
                        fooled_loss.item()
                    )

                    print(msg)

                    
                    nrows = int(math.sqrt(batch_size))

                    stamp = "{}-{}-{}".format(now, epoch_num, batch_num)

                    tensors_paths =[
                        (decoded, experiment_root + r"\Images\Output\{}-decoded.png".format(stamp)),
                        (fake_images, experiment_root + r"\Images\Output\{}-fake.png".format(stamp)),
                        # (data, experiment_root + r"\Images\Output\{}-real.png".format(stamp)),
                    ]

                    AEGeAN.save_images(tensors_paths, nrows)
            # Per Epoch
            torch.save(copy.deepcopy(D).cpu().state_dict(),
                experiment_root + r"\Models\{}-discriminator.pth".format(now))

            torch.save(copy.deepcopy(G).cpu().state_dict(),
                experiment_root + r"\Models\{}-generator.pth".format(now))


if __name__ == "__main__":
    experiment_root = r"D:\MLExperiments\AEGeAN"

    data_root = r"D:\Datasets\ALotOfPlansModified"

    dataloader, n_batches = AEGeAN.image_dataset(data_root, batch_size=9)

    models = {
        "generator": torch.load(experiment_root + r"\Models\Good\1561579878-generator.pth"),
        "discriminator": torch.load(experiment_root + r"\Models\Good\1561579878-discriminator.pth")
    }

    AEGeAN.train(dataloader, 1000, n_batches, experiment_root, models=models)