import torch
import torch.nn as nn
import timm

from config import CFG


class Encoder(nn.Module):

    def __init__(
        self,
        model_name=CFG.model_name,
        pretrained=True,
        out_dim=256
    ):
        super().__init__()

        self.model = timm.create_model(
            model_name,
            num_classes=0,
            global_pool="",
            pretrained=pretrained
        )

        self.bottleneck = nn.AdaptiveAvgPool1d(out_dim)

    def forward(self, x):

        features = self.model(x)

        # Remove CLS token
        features = features[:, 1:]

        # Convert feature dimension to the decoder dimension
        features = self.bottleneck(features)

        return features


class Decoder(nn.Module):

    def __init__(
        self,
        vocab_size=CFG.vocab_size,
        embed_dim=256,
        num_heads=8,
        num_layers=4,
        max_len=CFG.max_len
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embed_dim,
            padding_idx=CFG.pad_idx
        )

        self.pos_embedding = nn.Parameter(
            torch.randn(1, max_len, embed_dim)
        )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            batch_first=True
        )

        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers
        )

        self.fc_out = nn.Linear(
            embed_dim,
            vocab_size
        )

    def forward(self, tgt, memory):

        tgt_emb = self.embedding(tgt)

        seq_len = tgt_emb.size(1)

        tgt_emb = tgt_emb + self.pos_embedding[:, :seq_len]

        tgt_mask = nn.Transformer.generate_square_subsequent_mask(
            seq_len,
            device=tgt.device
        )

        tgt_padding_mask = (
            tgt == CFG.pad_idx
        )

        output = self.decoder(
            tgt=tgt_emb,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_padding_mask
        )

        output = self.fc_out(output)

        return output


class MDCNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.encoder = Encoder()

        self.decoder = Decoder()

    def forward(self, images, captions):

        memory = self.encoder(images)

        output = self.decoder(
            captions,
            memory
        )

        return output