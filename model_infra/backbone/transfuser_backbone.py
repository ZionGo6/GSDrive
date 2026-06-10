"""
Implements the TransFuser vision backbone.
"""

import copy
import math

import timm
import torch
import torch.nn.functional as F
from torch import nn

from model_infra.backbone.backbone_config import TransfuserConfig



class TransfuserBackbone(nn.Module):
    """Multi-scale Fusion Transformer."""

    def __init__(self, config: TransfuserConfig):

        super().__init__()
        self.config = config
        try:
            self.image_encoder = timm.create_model(config.image_architecture, pretrained=True, features_only=True)
        except Exception as e:
            print(f"Failed to load image encoder with error: {e}")
            self.image_encoder = timm.create_model(config.image_architecture, pretrained=True, features_only=True,
                                                   pretrained_cfg_overlay=dict(file=config.bkb_path))

        self.avgpool_img = nn.AdaptiveAvgPool2d((self.config.img_vert_anchors, self.config.img_horz_anchors))

        self.global_pool_img = nn.AdaptiveAvgPool2d(output_size=1)
        start_index = 0
        
        if len(self.image_encoder.return_layers) > 4:
            start_index += 1

        self.transformers = nn.ModuleList(
            [
                GPT(
                    n_embd=self.image_encoder.feature_info.info[start_index + i]["num_chs"],
                    config=config,
                )
                for i in range(4)
            ]
        )

        self.num_image_features = self.image_encoder.feature_info.info[start_index + 3]["num_chs"]

        self.perspective_upsample_factor = (
            self.image_encoder.feature_info.info[start_index + 3]["reduction"]
            // self.config.perspective_downsample_factor
        )

        # Number of features the encoder produces.
        self.num_features = self.image_encoder.feature_info.info[start_index + 3]["num_chs"]

    def forward(self, image):
        """
        Image feature processing using transformers
        Args:
            image: input images
        """

        b, f, c, h, w = image.shape
        image_features = image.reshape(b * f, c, h, w)

        # Generate an iterator for all the layers in the network that one can loop through.
        image_layers = iter(self.image_encoder.items())

        if len(self.image_encoder.return_layers) > 4:
            image_features = self.forward_layer_block(image_layers, self.image_encoder.return_layers, image_features)

        for i in range(4):
            image_features = self.forward_layer_block(image_layers, self.image_encoder.return_layers, image_features)
            image_features = self.process_image_features(image_features, i)

        image_features = self.global_pool_img(image_features)
        flatten_features = torch.flatten(image_features, 1)

        return flatten_features

    def forward_layer_block(self, layers, return_layers, features):
        """
        Run one forward pass to a block of layers from a TIMM neural network and returns the result.
        Advances the whole network by just one block
        :param layers: Iterator starting at the current layer block
        :param return_layers: TIMM dictionary describing at which intermediate layers features are returned.
        :param features: Input features
        :return: Processed features
        """
        for name, module in layers:
            features = module(features)
            if name in return_layers:
                break
        return features

    def process_image_features(self, image_features, layer_idx):
        """
        Process image features using a Transformer module.
        :param image_features: Features from the image branch
        :param layer_idx: Transformer layer index.
        :return: image_features with added features from transformer.
        """
        image_embd_layer = self.avgpool_img(image_features)

        image_features_layer = self.transformers[layer_idx](image_embd_layer)

        image_features_layer = F.interpolate(
            image_features_layer,
            size=(image_features.shape[2], image_features.shape[3]),
            mode="bilinear",
            align_corners=False,
        )

        image_features = image_features + image_features_layer

        return image_features


class GPT(nn.Module):
    """The full GPT language backbone, with a context size of block_size."""

    def __init__(self, n_embd, config):
        super().__init__()
        self.n_embd = n_embd
        self.seq_len = 1
        self.config = config

        self.pos_emb = nn.Parameter(
            torch.zeros(
                1,
                self.seq_len * self.config.img_vert_anchors * self.config.img_horz_anchors,
                self.n_embd,
            )
        )

        self.drop = nn.Dropout(config.embd_pdrop)

        # transformer
        self.blocks = nn.Sequential(
            *[
                Block(n_embd, config.n_head, config.block_exp, config.attn_pdrop, config.resid_pdrop)
                for layer in range(config.n_layer)
            ]
        )

        # decoder head
        self.ln_f = nn.LayerNorm(n_embd)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(
                mean=self.config.gpt_linear_layer_init_mean,
                std=self.config.gpt_linear_layer_init_std,
            )
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(self.config.gpt_layer_norm_init_weight)

    def forward(self, image_tensor):
        """
        Args:
            image_tensor (tensor): B*4*seq_len, C, H, W
        """

        bz = image_tensor.shape[0]
        img_h, img_w = image_tensor.shape[2:4]

        assert self.seq_len == 1
        image_tensor = image_tensor.permute(0, 2, 3, 1).contiguous().view(bz, -1, self.n_embd)

        token_embeddings = image_tensor

        x = self.drop(self.pos_emb + token_embeddings)
        x = self.blocks(x)  # (B, an * T, C)
        x = self.ln_f(x)  # (B, an * T, C)

        image_tensor_out = (
            x[:, : self.seq_len * self.config.img_vert_anchors * self.config.img_horz_anchors, :]
            .view(bz * self.seq_len, img_h, img_w, -1)
            .permute(0, 3, 1, 2)
            .contiguous()
        )

        return image_tensor_out


class SelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the
    end.
    """

    def __init__(self, n_embd, n_head, attn_pdrop, resid_pdrop):
        super().__init__()
        assert n_embd % n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)
        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        # output projection
        self.proj = nn.Linear(n_embd, n_embd)
        self.n_head = n_head

    def forward(self, x):
        b, t, c = x.size()

        # calculate query, key, values for all heads in batch and move head
        # forward to be the batch dim
        k = self.key(x).view(b, t, self.n_head, c // self.n_head).transpose(1, 2)  # (b, nh, t, hs)
        q = self.query(x).view(b, t, self.n_head, c // self.n_head).transpose(1, 2)  # (b, nh, t, hs)
        v = self.value(x).view(b, t, self.n_head, c // self.n_head).transpose(1, 2)  # (b, nh, t, hs)

        # self-attend: (b, nh, t, hs) x (b, nh, hs, t) -> (b, nh, t, t)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v  # (b, nh, t, t) x (b, nh, t, hs) -> (b, nh, t, hs)
        y = y.transpose(1, 2).contiguous().view(b, t, c)  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y


class Block(nn.Module):
    """an unassuming Transformer block"""

    def __init__(self, n_embd, n_head, block_exp, attn_pdrop, resid_pdrop):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.attn = SelfAttention(n_embd, n_head, attn_pdrop, resid_pdrop)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, block_exp * n_embd),
            nn.ReLU(True),  # changed from GELU
            nn.Linear(block_exp * n_embd, n_embd),
            nn.Dropout(resid_pdrop),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))

        return x


class MultiheadAttentionWithAttention(nn.Module):
    """
    MultiheadAttention that also return attention weights
    """

    def __init__(self, n_embd, n_head, pdrop):
        super().__init__()
        assert n_embd % n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)
        # regularization
        self.attn_drop = nn.Dropout(pdrop)
        self.resid_drop = nn.Dropout(pdrop)
        # output projection
        self.proj = nn.Linear(n_embd, n_embd)
        self.n_head = n_head

    def forward(self, q_in, k_in, v_in):
        b, t, c = q_in.size()
        _, t_mem, _ = k_in.size()

        # calculate query, key, values for all heads in batch and move head
        # forward to be the batch dim
        q = self.query(q_in).view(b, t, self.n_head, c // self.n_head).transpose(1, 2)  # (b, nh, t, hs)
        k = self.key(k_in).view(b, t_mem, self.n_head, c // self.n_head).transpose(1, 2)  # (b, nh, t, hs)
        v = self.value(v_in).view(b, t_mem, self.n_head, c // self.n_head).transpose(1, 2)  # (b, nh, t, hs)

        # self-attend: (b, nh, t, hs) x (b, nh, hs, t) -> (b, nh, t, t)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v  # (b, nh, t, t) x (b, nh, t, hs) -> (b, nh, t, hs)
        y = y.transpose(1, 2).contiguous().view(b, t, c)  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        attention = torch.mean(att, dim=1)  # Average attention over heads
        return y, attention


class TransformerDecoderLayerWithAttention(nn.Module):
    """A Transformer decoder that returns the attentions."""

    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation=F.relu,
        layer_norm_eps=1e-5,
    ):
        super().__init__()
        self.self_attn = MultiheadAttentionWithAttention(d_model, nhead, dropout)
        self.multihead_attn = MultiheadAttentionWithAttention(d_model, nhead, dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = activation

    def forward(self, tgt, memory):
        x = tgt
        tmp, _ = self.self_attn(x, x, x)
        x = self.norm1(x + self.dropout1(tmp))
        tmp, attention = self.multihead_attn(x, memory, memory)
        x = self.norm2(x + self.dropout2(tmp))
        tmp = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = self.norm3(x + self.dropout3(tmp))

        return x, attention


class TransformerDecoderWithAttention(nn.Module):
    """A Transformer decoder that returns the attentions."""

    def __init__(self, layers, num_layers, norm=None):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layers) for i in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, queries, memory):
        output = queries
        attentions = []
        for mod in self.layers:
            output, attention = mod(output, memory)
            attentions.append(attention)

        if self.norm is not None:
            output = self.norm(output)

        avg_attention = torch.mean(torch.stack(attentions), dim=0)
        return output, avg_attention
