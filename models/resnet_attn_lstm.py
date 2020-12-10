import torch
from torch.nn import functional as F
from torch import nn
from pytorch_lightning.core.lightning import LightningModule
from torchvision.models import resnet50, resnet152, densenet161
from models.models import ModelsManager

from models.resnet import ResNet50


@ModelsManager.export("Mnist")
class LitMNIST(LightningModule):
    def __init__(self, args):
        super().__init__()

        # mnist images are (1, 28, 28) (channels, width, height)
        self.layer_1 = torch.nn.Linear(28 * 28, 128)
        self.layer_2 = torch.nn.Linear(128, 256)
        self.layer_3 = torch.nn.Linear(256, 10)

    def forward(self, x):
        batch_size, channels, width, height = x.size()

        # (b, 1, 28, 28) -> (b, 1*28*28)
        x = x.view(batch_size, -1)
        x = self.layer_1(x)
        x = F.relu(x)
        x = self.layer_2(x)
        x = F.relu(x)
        x = self.layer_3(x)

        x = F.log_softmax(x, dim=1)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        return loss

class BahdanauAttention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        """
        encoder_dim: feature size of encoded images
        decoder_dim: size of decoder's RNN
        attention_dim: size of the attention network
        """
        super(BahdanauAttention, self).__init__()
        self.w_enc = nn.Linear(encoder_dim, attention_dim)   # linear layer to transform encoded image
        self.w_dec = nn.Linear(decoder_dim, attention_dim)   # linear layer to transform decoder's output 
        self.full_att = nn.Linear(attention_dim, 1)  # linear layer to calculate values to be softmax-ed
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)  # softmax layer to calculate weights

    def forward(self, encoder_out, decoder_hidden):
        """
        Forward propagation.
        encoder_out: encoded images, a tensor of dimension (batch_size, num_channels, encoder_dim)
        decoder_hidden: previous decoder output, a tensor of dimension (batch_size, decoder_dim)
        attention weighted encoding, weights
        """
        decoder_hidden_with_time_axis = torch.unsqueeze(decoder_hidden, 1) #(batch_size, 1, decoder_dim)
        attention_hidden_layer = self.tanh(self.w_enc(encoder_out) + self.w_dec(decoder_hidden_with_time_axis))  # (batch_size,channel, attention_dim)
        attention_weights = self.softmax(self.full_att(attention_hidden_layer).squeeze(2))  # (batch_size, channels)
        attention_weighted_encoding = (encoder_out * attention_weights.unsqueeze(2)).sum(dim=1)  # (batch_size, encoder_dim)

        return attention_weighted_encoding, attention_weights

class Encoder(nn.Module):
    def __init__(self, network='resnet152', pretrained = True):
        super(Encoder, self).__init__()
        self.network = network
        if network == 'resnet152':
            self.net = resnet152(pretrained=pretrained)
            self.net = nn.Sequential(*list(self.net.children())[:-2])
            self.dim = 2048
        elif network == 'densenet161':
            self.net = densenet161(pretrained=pretrained)
            self.net = nn.Sequential(*list(list(self.net.children())[0])[:-1])
            self.dim = 1920
        else:
            self.net = resnet50(pretrained=pretrained)
            self.net = nn.Sequential(*list(self.net.features.children())[:-1])
            self.dim = 2048

    def forward(self, x):
        x = self.net(x)
        # x = x.permute(0, 2, 3, 1)
        x = x.view(x.size(0), -1, x.size(-1))
        return x