import torch
import torch.nn as nn
import timm

from src.config import CFG


class Encoder(nn.Module):
    def __init__(self, model_name=CFG.model_name, pretrained=True, out_dim=256):
        super().__init__()

        self.model = timm.create_model(
            model_name,
            num_classes=0,
            global_pool="",
            pretrained=pretrained
        )

        # timm DeiT returns token features [B, tokens, encoder_dim].
        encoder_dim = self.model.num_features
        self.feature_projection = nn.Linear(encoder_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)

        self.classifier = nn.Sequential(
            nn.LayerNorm(out_dim),
            nn.Dropout(0.2),
            nn.Linear(out_dim, CFG.num_classes)
        )
        self.location_classifier = nn.Sequential(
            nn.LayerNorm(out_dim),
            nn.Dropout(0.2),
            nn.Linear(out_dim, CFG.num_locations)
        )

        # Predict normalized [xmin, ymin, xmax, ymax] from image features.
        self.bbox_regressor = nn.Sequential(
            nn.LayerNorm(out_dim),
            nn.Linear(out_dim, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 4),
            nn.Sigmoid()
        )

    def forward(self, x, bboxes=None):
        tokens = self.model.forward_features(x)

        # Some timm architectures return a tensor of patch tokens.
        if tokens.ndim == 4:
            # [B, C, H, W] -> [B, H*W, C]
            tokens = tokens.flatten(2).transpose(1, 2)

        if tokens.ndim != 3:
            raise RuntimeError(
                f"Expected 3-D token features from visual encoder, got {tuple(tokens.shape)}"
            )

        # Drop CLS token when present (196 patches + CLS for 224/16 ViT).
        if tokens.size(1) == CFG.num_patches + 1:
            patch_tokens = tokens[:, 1:, :]
        else:
            patch_tokens = tokens

        patch_tokens = self.norm(self.feature_projection(patch_tokens))
        global_features = patch_tokens.mean(dim=1)

        class_logits = self.classifier(global_features)
        location_logits = self.location_classifier(global_features)
        bbox_pred = self.bbox_regressor(global_features)

        # Ensure x_min <= x_max and y_min <= y_max.
        x1 = torch.minimum(bbox_pred[:, 0], bbox_pred[:, 2])
        y1 = torch.minimum(bbox_pred[:, 1], bbox_pred[:, 3])
        x2 = torch.maximum(bbox_pred[:, 0], bbox_pred[:, 2])
        y2 = torch.maximum(bbox_pred[:, 1], bbox_pred[:, 3])
        bbox_pred = torch.stack((x1, y1, x2, y2), dim=1)

        # Decoder attends to image patch features only. Do not feed the
        # ground-truth box into inference; that would leak the answer.
        return patch_tokens, class_logits, location_logits, bbox_pred


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
            vocab_size, embed_dim, padding_idx=CFG.pad_idx
        )
        self.pos_embedding = nn.Parameter(
            torch.randn(1, max_len, embed_dim) * 0.02
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            batch_first=True,
            norm_first=True
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=num_layers,
            norm=nn.LayerNorm(embed_dim)
        )
        self.fc_out = nn.Linear(embed_dim, vocab_size)

    def forward(self, tgt, memory):
        tgt_emb = self.embedding(tgt)
        seq_len = tgt_emb.size(1)
        if seq_len > self.pos_embedding.size(1):
            raise ValueError(
                f"Caption length {seq_len} exceeds max_len={self.pos_embedding.size(1)}"
            )
        tgt_emb = tgt_emb + self.pos_embedding[:, :seq_len]

        # True entries mask future positions.
        tgt_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=tgt.device, dtype=torch.bool),
            diagonal=1
        )
        tgt_padding_mask = tgt.eq(CFG.pad_idx)
        decoded = self.decoder(
            tgt=tgt_emb,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_padding_mask
        )
        return self.fc_out(decoded)


class MDCNet(nn.Module):
    def __init__(self, vocab_size=None):
        super().__init__()
        self.encoder = Encoder()
        # Use the vocabulary built from the training captions when provided.
        # Fall back to CFG.vocab_size only for older callers.
        resolved_vocab_size = CFG.vocab_size if vocab_size is None else int(vocab_size)
        self.decoder = Decoder(vocab_size=resolved_vocab_size)

    def forward(self, images, captions, bboxes=None):
        memory, class_logits, location_logits, bbox_pred = self.encoder(images)
        caption_logits = self.decoder(captions, memory)
        return caption_logits, class_logits, location_logits, bbox_pred
