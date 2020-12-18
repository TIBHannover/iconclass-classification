from torch import nn

class BahdanauAttention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        """
        encoder_dim: feature size of encoded images
        decoder_dim: size of decoder's RNN
        attention_dim: size of the attention network
        """
        super(BahdanauAttention, self).__init__()
        self.w_enc = nn.Linear(encoder_dim, attention_dim)  # linear layer to transform encoded image
        self.w_dec = nn.Linear(decoder_dim, attention_dim)  # linear layer to transform decoder's output
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
        decoder_hidden_with_time_axis = torch.unsqueeze(decoder_hidden, 1)  # (batch_size, 1, decoder_dim)
        attention_hidden_layer = self.tanh(
            self.w_enc(encoder_out) + self.w_dec(decoder_hidden_with_time_axis)
        )  # (batch_size,channel, attention_dim)
        attention_weights = self.softmax(self.full_att(attention_hidden_layer).squeeze(2))  # (batch_size, channels)
        attention_weighted_encoding = (encoder_out * attention_weights.unsqueeze(2)).sum(
            dim=1
        )  # (batch_size, encoder_dim)

        return attention_weighted_encoding, attention_weights
