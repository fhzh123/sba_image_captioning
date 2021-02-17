# coding: utf-8
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .embedding.positional import SpatialPositionalEmbedding
from .embedding.transformer_embedding import TransformerEmbedding

class TransformerCaptioning(nn.Module):
    def __init__(self, vocab_num, pad_idx=0, bos_idx=1, eos_idx=2, max_len=100, feature_dim=2048, 
        feature_size=14, d_model=512, n_head=8, dim_feedforward=2048, num_encoder_layer=6, 
        num_decoder_layer=6, dropout=0.1, embedding_dropout=0.1):
        
        super(TransformerCaptioning, self).__init__()

        # Model Hyper-parameter Setting
        self.pad_idx = pad_idx
        self.bos_idx = bos_idx
        self.eos_idx = eos_idx
        self.max_len = max_len
        self.d_model = d_model
        
        # EfficientNet Hyper-parameter Setting
        self.extractor = None
        self.extractor_dim = feature_dim
        self.feature_size = feature_size

        # Image Processing Model Setting
        self.img_feature_linear = nn.Linear(self.extractor_dim, d_model, bias=False)
        self.image_positional_embedding = SpatialPositionalEmbedding(d_model=d_model, 
                                            x_max_len=feature_size, y_max_len=feature_size)
        self.image_dropout = nn.Dropout(embedding_dropout)

        # Caption Processing Model Setting
        self.embedding = TransformerEmbedding(vocab_num, d_model, pad_idx=self.pad_idx, 
                            max_len=self.max_len, embedding_dropout=embedding_dropout)
        self.transformer = nn.Transformer(d_model, n_head, num_encoder_layer, num_decoder_layer, 
                            dim_feedforward, dropout, activation='gelu')
        self.output_linear = nn.Linear(d_model, vocab_num, bias=False)
        
        # Regularization Model Setting
        self.img_layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, img, dec_input_sentence, tgt_key_padding_mask, tgt_mask):
        dec_embs = self.embedding(dec_input_sentence)
        dec_embs = torch.einsum('ijk->jik', dec_embs)
        extractor_out = self._extractor_out(img)

        outputs = self.transformer(extractor_out, dec_embs,
                        tgt_key_padding_mask=tgt_key_padding_mask, 
                        tgt_mask=tgt_mask)

        outputs = torch.einsum('ijk->jik', outputs)
        outputs = self.output_linear(outputs)
        return outputs

    def _extractor_out(self, img):
        with torch.no_grad():
            # B * dim * W * H
            img_feature = self.extractor.extract_features(img)
            
        # B * W * H * dim 
        img_feature = img_feature.permute(0, 2, 3, 1)
        img_feature = self.image_dropout(F.gelu(self.img_feature_linear(img_feature)))
        # B * W*H * model_dim
        img_feature = img_feature.view(img.size(0), -1, self.d_model)
        img_feature += self.image_positional_embedding(img_feature)
        img_feature = self.img_layer_norm(img_feature)
        
        # W*H * B * model_dim
        img_feature = torch.einsum('ijk->jik', img_feature)
        return img_feature

    def predict(self, img, device):
        predicted = torch.LongTensor([[self.bos_idx]]).to(device)
        extractor_out = self._extractor_out(img)

        for _ in range(self.max_len):
            dec_embs = self.embedding(predicted)
            outputs = self.transformer(extractor_out, dec_embs)
            outputs = torch.einsum('ijk->jik', outputs)
            outputs = self.output_linear(outputs)
            y_pred_id = outputs.max(dim=2)[1][-1, 0]

            if y_pred_id == self.eos_idx:
                break

            predicted = torch.cat([predicted, y_pred_id.view(1, 1)], dim=0)
            
        predicted = predicted[1:, 0].cpu().numpy() # remove bos token
        return predicted