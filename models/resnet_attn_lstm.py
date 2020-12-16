import argparse

import torch
from torch.nn import functional as F
from torch import nn
from pytorch_lightning.core.lightning import LightningModule
from torchvision.models import resnet50, resnet152, densenet161, inception_v3
from models.models import ModelsManager

from models.resnet import ResNet50

from datasets.utils import read_jsonl


@ModelsManager.export("attn_lstm")
class AttnLstm(LightningModule):
    def __init__(self, args=None, **kwargs):
        super(AttnLstm, self).__init__()
        if args is not None:
            dict_args = vars(args)
            dict_args.update(kwargs)
        else:
            dict_args = kwargs

        self.encode_model = dict_args.get("encode_model", None)
        self.pretrained = dict_args.get("pretrained", None)

        self.mapping_path = dict_args.get("mapping_path", None)
        self.classifier_path = dict_args.get("classifier_path", None)
        self.mapping = {}
        if self.mapping_path is not None:
            self.mapping = read_jsonl(self.mapping_path, dict_key="id")

        self.classifier = {}
        if self.classifier_path is not None:
            self.classifier = read_jsonl(self.classifier_path)
            
        self.max_level = len(self.classifier)
                    
        self.vocabulary_size = [len(x['tokenizer']) for x in self.classifier] #get from tockenizer
        self.max_vocab_size = max(self.vocabulary_size)
        self.embedding_dim = 128
        self.attention_dim = 64
        #self.max_vocab_size = max(self.vocabulary_size)
        self.encoder = Encoder(network='resnet152', embedding_dim = self.embedding_dim, pretrained = True)
        self.encoder_dim = self.encoder.dim
        self.decoder = Decoder( self.vocabulary_size, self.embedding_dim, self.attention_dim, self.embedding_dim, self.max_vocab_size)
        self.loss = torch.nn.BCEWithLogitsLoss()

    def forward(self, x):
        
        return x

    def training_step(self, batch, batch_idx):
        image = batch["image"]
        source = batch["source_id_sequnce"]
        target = batch["target_vec"]
        # image = F.interpolate(image, size = (299,299), mode= 'bicubic', align_corners=False)
        # print(image.shape)
        # forward image
        image_embedding = self.encoder(image)
        # print('*********')
        # print(image_embedding.shape)
        # return loss
        hidden = self.decoder.reset_state(image.shape[0]).to(image.device.index)
        # print(hidden.device)
        
        # Feed <START> to the model in the first layer 1==<START>
        decoder_inp = torch.ones([image.shape[0],1],dtype=torch.int64).to(image.device.index)
        # print('#########################')
        # print(decoder_inp)
        loss = 0
        for i_lev in range(len(target)):    
            predictions, hidden, _ = self.decoder(decoder_inp, image_embedding, hidden, i_lev)
            #print('#########################')
            #print(predictions.shape)
            #print('#########################')
            #print(target[i_lev].shape)
            loss += self.loss(predictions,target[i_lev])
            decoder_inp = torch.unsqueeze(source[i_lev], dim=1)
            #print(decoder_inp.shape)
            #decoder_inp = torch.tensor(target[i_lev]).to(torch.int64).to(image.device.index)
        # print('#########################')
        # print(predictions.shape)
        total_loss = loss/len(target)
                
        #print(torch.mean(loss))
        loss = total_loss
        return {'loss':torch.mean(loss)}
    
    def training_step_end(self, outputs):
        self.log('train/loss', outputs['loss'].mean(), prog_bar=True)
        
        return {'loss': outputs['loss'].mean()}
    
    
    def validation_step(self, batch, batch_idx):
        image = batch["image"]
        source = batch["source_id_sequnce"]
        target = batch["target_vec"]
        # image = F.interpolate(image, size = (299,299), mode= 'bicubic', align_corners=False)
        # print(image.shape)
        # forward image
        image_embedding = self.encoder(image)
        # print('*********')
        # print(image_embedding.shape)
        # return loss
        hidden = self.decoder.reset_state(image.shape[0]).to(image.device.index)
        # print(hidden.device)
        
        # Feed <START> to the model in the first layer 1==<START>
        decoder_inp = torch.ones([image.shape[0],1],dtype=torch.int64).to(image.device.index)
        # print('#########################')
        # print(decoder_inp)
        loss = 0
        for i_lev in range(len(target)):    
            predictions, hidden, _ = self.decoder(decoder_inp, image_embedding, hidden, i_lev)

            loss += self.loss(predictions,target[i_lev])
            decoder_inp = torch.unsqueeze(source[i_lev], dim=1)

            #decoder_inp = torch.tensor(target[i_lev]).to(torch.int64).to(image.device.index)
        # print('#########################')
        # print(predictions.shape)
        total_loss = loss/len(target)
                
        return {'loss':torch.mean(loss)}
    
    def validation_epoch_end(self, outputs):

        loss = 0.0
        count = 0
        for output in outputs:
            loss += output["loss"]
            count += 1

        self.log('val/loss', loss / count, prog_bar=True)
        return {
            "loss": loss / count,
        }
        
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(params=list(self.encoder.parameters())+ list(self.decoder.parameters()),
        lr=0.001,weight_decay=0.01)
        return optimizer
        
    @classmethod
    def add_args(cls, parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False, conflict_handler="resolve")
        # args, _ = parser.parse_known_args()
        # if "classifier_path" not in args:
        parser.add_argument("--pretrained", type=bool, default=True)
        parser.add_argument("--encode_model", type=str, default="resnet50")
        parser.add_argument("--mapping_path", type=str)
        parser.add_argument("--classifier_path", type=str)
        return parser

class BahdanauAttention(nn.Module):
    def __init__(self, encoder_dim,attention_dim):
        """
        encoder_dim: feature size of encoded images
        decoder_dim: size of decoder's RNN
        """
        super(BahdanauAttention, self).__init__()
        self.w_enc = nn.Linear(encoder_dim, attention_dim)   # linear layer to transform encoded image
        self.w_dec = nn.Linear(attention_dim, attention_dim)   # linear layer to transform decoder's output 
        self.full_att = nn.Linear(attention_dim, 1)  # linear layer to calculate values to be softmax-ed
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)  # softmax layer to calculate weights

    def forward(self, encoder_out, decoder_hidden):
        """
        Forward propagation.
        encoder_out: encoded images, a tensor of dimension (batch_size, num_channels, encoder_dim)
        """
        decoder_hidden_with_time_axis = torch.unsqueeze(decoder_hidden, 1) #(batch_size, 1, decoder_dim)
        # print('decoder_hidden_with_time_axis.shape {}'.format(decoder_hidden_with_time_axis.shape))
        # print('encoder_out.shape{}'.format(encoder_out.shape))
        # print('hidden {}'.format(decoder_hidden.shape))
        
        attention_hidden_layer = self.tanh(self.w_enc(encoder_out) + self.w_dec(decoder_hidden_with_time_axis))  # (batch_size,channel, attention_dim)
        # print('score {}'.format(attention_hidden_layer.shape))
        attention_weights = self.softmax(self.full_att(attention_hidden_layer).squeeze(2))  # (batch_size, channels)
        
        attention_weighted_encoding = (encoder_out * attention_weights.unsqueeze(2)).sum(dim=1)  # (batch_size, encoder_dim)
        # print('context vector {}'.format(attention_weighted_encoding.shape))
        # print('attntion weights {}'.format(attention_weights.shape))
        return attention_weighted_encoding, attention_weights

class Encoder(nn.Module):
    def __init__(self, network='resnet152', embedding_dim = 128, pretrained = True):
        super(Encoder, self).__init__()
        self.network = network
        if network == 'resnet152':
            self.net = resnet152(pretrained=pretrained, progress=False)
            self.net = nn.Sequential(*list(self.net.children())[:-2])
            self.dim = 2048
        elif network == 'densenet161':
            self.net = densenet161(pretrained=pretrained)
            self.net = nn.Sequential(*list(list(self.net.children())[0])[:-1])
            self.dim = 1920
        elif network == 'resnet50':
            self.net = resnet50(pretrained=pretrained)
            self.net = nn.Sequential(*list(self.net.features.children())[:-1])
            self.dim = 2048
        elif network == 'inceptionv3': # TODO:: fix the input dimension of images
            self.net = inception_v3(pretrained = pretrained, progress = False)
            self.net = nn.Sequential(*list(self.net.children())[:-1])
            self.dim = 2048
        self.embedding_dim = embedding_dim
        self._fc = nn.Linear(self.dim, self.embedding_dim) # Todo add layers 
        self._conv1 = torch.nn.Conv2d(self.dim, self.embedding_dim, kernel_size=[1, 1])
    def forward(self, x):
        x = self.net(x)
        x = self._conv1(x)
        x = x.permute(0, 2, 3, 1)
        x = x.view(x.size(0), -1, x.size(-1))
        return x
    
class Decoder(nn.Module):
    def __init__(self, vocabulary_size, embedding_dim, attention_dim,encoder_dim, max_vocab_size, tf=False):
        super(Decoder, self).__init__()
        self.attention_dim = attention_dim
        self.embedding = nn.ModuleList([nn.Embedding(max_vocab_size, embedding_dim) for x in range(len(vocabulary_size))])
        self.gru = nn.GRU(embedding_dim + encoder_dim, attention_dim, bias= True)
        # Todo add more linear layers Todo add kernel regulizer
        self.fc1 = nn.Linear(attention_dim, attention_dim)
        self.fc2 = nn.ModuleList([nn.Linear(attention_dim, attention_dim) for i in range(len(vocabulary_size))])
        self.fc3 = nn.ModuleList([nn.Linear(attention_dim, vocabulary_size[i]) for i in range(len(vocabulary_size))])
        self.attention = BahdanauAttention(encoder_dim, attention_dim)
        #Todo add drop out layers
    
    def forward(self, x, encoder_out, hidden, level):
        """
        Forward propagation.
        encoder_out: encoded images, a tensor of dimension (batch_size, enc_image_size, enc_image_size, encoder_dim)
        decoder_inp: encoded captions, a tensor of dimension (batch_size, max_caption_length)
        #caption_lengths: caption lengths, a tensor of dimension (batch_size, 1)
        return: scores for vocabulary, sorted encoded captions, decode lengths, weights, sort indices
        """
        context_vec, attention_weights = self.attention(encoder_out, hidden)
        
        # print(level)
        # print(x.shape)
        if level<9:
            x = self.embedding[level](x)
        #else:??
        #    x = self.embedding[7](decoder_inp)
        # print(x.shape)
        # print(context_vec.shape)
        x = torch.cat([context_vec.unsqueeze(1), x], dim=2)
        # print('before gru {}'.format(x.shape))
        output, state = self.gru(x.permute(1, 0, 2))
        # print('output after gru {}'.format(output.shape))
        
        output = output.permute(1, 0, 2)
        state =  state.squeeze(0)
        # print('state after gru {}'.format(state.shape))
        x = self.fc1(output)
        x = torch.reshape(x, (-1, x.size()[2]))
        # print('after gru reshape {}'.format(x.shape))
        if level<9 :
            x = self.fc2[level](x)
            x = self.fc3[level](x)
        #else:???
        #    x = self.fc2[7](x)
        #    x = self.fc3[7](x)
        # print('output of decoder{}'.format(x.shape))
        return x, state, attention_weights
        
    def reset_state(self, batch_size):
        return torch.zeros((batch_size, self.attention_dim))
