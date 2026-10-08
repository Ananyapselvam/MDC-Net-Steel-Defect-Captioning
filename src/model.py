import torch
import torch.nn as nn
import timm

from src.config import CFG


class Encoder(nn.Module):

    def __init__(
        self,
        model_name=CFG.model_name,
        pretrained=True,
        out_dim=256
    ):
        super().__init__()

        # --------------------------------------------------
        # Visual encoder: DeiT3
        # --------------------------------------------------
        self.model = timm.create_model(
            model_name,
            num_classes=0,
            global_pool="",
            pretrained=pretrained
        )

        # Convert visual features to decoder dimension
        self.bottleneck = nn.AdaptiveAvgPool1d(out_dim)

        # --------------------------------------------------
        # Defect classification head
        # 6 defect classes
        # --------------------------------------------------
        self.classifier = nn.Linear(
            out_dim,
            CFG.num_classes
        )

        # --------------------------------------------------
        # Spatial location classification head
        # 9 spatial classes
        # --------------------------------------------------
        self.location_classifier = nn.Linear(
            out_dim,
            CFG.num_locations
        )

        # --------------------------------------------------
        # Bounding-box projection
        #
        # bbox = [xmin, ymin, xmax, ymax]
        # normalized to [0, 1]
        # --------------------------------------------------
        self.bbox_projection = nn.Sequential(
            nn.Linear(4, 128),
            nn.ReLU(),
            nn.Linear(128, out_dim)
        )

    def forward(
        self,
        x,
        bboxes
    ):

        # --------------------------------------------------
        # Extract visual features
        # --------------------------------------------------
        features = self.model(x)

        # Remove CLS token
        features = features[:, 1:]

        # Convert feature dimension
        features = self.bottleneck(features)

        # --------------------------------------------------
        # Global visual representation
        # --------------------------------------------------
        global_features = features.mean(
            dim=1
        )

        # --------------------------------------------------
        # Defect classification
        # --------------------------------------------------
        class_logits = self.classifier(
            global_features
        )

        # --------------------------------------------------
        # Location classification
        # --------------------------------------------------
        location_logits = (
            self.location_classifier(
                global_features
            )
        )

        # --------------------------------------------------
        # Convert bbox into location token
        # --------------------------------------------------
        bbox_features = self.bbox_projection(
            bboxes
        )

        bbox_features = bbox_features.unsqueeze(1)

        # --------------------------------------------------
        # Append bbox information to visual memory
        # --------------------------------------------------
        features = torch.cat(
            [
                features,
                bbox_features
            ],
            dim=1
        )

        return (
            features,
            class_logits,
            location_logits
        )


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

        # --------------------------------------------------
        # Token embedding
        # --------------------------------------------------
        self.embedding = nn.Embedding(
            vocab_size,
            embed_dim,
            padding_idx=CFG.pad_idx
        )

        # --------------------------------------------------
        # Positional embedding
        # --------------------------------------------------
        self.pos_embedding = nn.Parameter(
            torch.randn(
                1,
                max_len,
                embed_dim
            )
        )

        # --------------------------------------------------
        # Transformer decoder
        # --------------------------------------------------
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            batch_first=True
        )

        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers
        )

        # --------------------------------------------------
        # Vocabulary prediction
        # --------------------------------------------------
        self.fc_out = nn.Linear(
            embed_dim,
            vocab_size
        )

    def forward(
        self,
        tgt,
        memory
    ):

        # --------------------------------------------------
        # Token embeddings
        # --------------------------------------------------
        tgt_emb = self.embedding(
            tgt
        )

        seq_len = tgt_emb.size(1)

        # Add positional embeddings
        tgt_emb = (
            tgt_emb
            + self.pos_embedding[
                :, :seq_len
            ]
        )

        # --------------------------------------------------
        # Causal mask
        # --------------------------------------------------
        tgt_mask = torch.triu(
        torch.ones(
            seq_len,
            seq_len,
            device=tgt.device,
            dtype=torch.bool
        ),
            diagonal=1
        )

        # Ignore padding tokens
        tgt_padding_mask = (
            tgt == CFG.pad_idx
        )

        # --------------------------------------------------
        # Transformer decoding
        # --------------------------------------------------
        output = self.decoder(
            tgt=tgt_emb,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_padding_mask
        )

        # --------------------------------------------------
        # Vocabulary logits
        # --------------------------------------------------
        output = self.fc_out(
            output
        )

        return output


class MDCNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.encoder = Encoder()

        self.decoder = Decoder()

    def forward(
        self,
        images,
        captions,
        bboxes
    ):

        # --------------------------------------------------
        # Encode image + bbox
        # --------------------------------------------------
        (
            memory,
            class_logits,
            location_logits
        ) = self.encoder(
            images,
            bboxes
        )

        # --------------------------------------------------
        # Generate caption
        # --------------------------------------------------
        caption_output = self.decoder(
            captions,
            memory
        )

        return (
            caption_output,
            class_logits,
            location_logits
        )