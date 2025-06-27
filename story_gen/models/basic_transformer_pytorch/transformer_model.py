# Author: Yuki Rivera
# This python file defines the transformer model class used in a separate 
# jupyter notebook that trains a basic transformer model


import torch
import torch.nn as nn


class TransformerModel(nn.Module): 
    def __init__(self, vocab_size, d_model=256, nhead=8, num_layers=2, dim_feedforward=512, dropout=0.1, max_len=50):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = self._build_positional_encoding(max_len, d_model)

        # encoder-decoder layer with multi-head self-attention, feedforward network with layernorm + dropout
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)

        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers)

        self.fc_out = nn.Linear(d_model, vocab_size)

    # Creates sine/cosine positional encodings 
    def _build_positional_encoding(self, max_len, d_model):

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
        # fills even-numbered dimensions with sin()
        pe[:, 0::2] = torch.sin(position * div_term)
        # fills odd-numbered dimensions with cos()
        pe[:, 1::2] = torch.cos(position * div_term)

        return pe.unsqueeze(0)  # shape: (1, max_len, d_model)


    def forward(self, src, tgt):
        # embeds the prompt/target using learned embeddings and adds positional encoding for each token's position
        src_embed = self.embedding(src) + self.positional_encoding[:, :src.size(1), :].to(src.device)
        tgt_embed = self.embedding(tgt) + self.positional_encoding[:, :tgt.size(1), :].to(tgt.device)

        # feeds the prompt/target into the transformer encoder/decoder
        memory = self.encoder(src_embed)
        output = self.decoder(tgt_embed, memory)
        
        # applies a final linear layer to convert decoder outputs into logits over the vocab
        return self.fc_out(output)