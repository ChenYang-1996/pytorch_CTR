import numpy as np
import torch
import torch.nn.functional as F
from .activation import get_activation_fn


class MultiheadAttentionInnerProduct(torch.nn.Module):

    def __init__(self, num_fields, embed_dim, num_heads, dropout):
        super().__init__()
        self.num_fields = num_fields
        self.mask = (torch.triu(torch.ones(num_fields, num_fields), diagonal=1) == 1)
        self.num_cross_terms = num_fields * (num_fields - 1) // 2
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout_p = dropout
        head_dim = embed_dim // num_heads
        assert head_dim * num_heads == embed_dim, "head dim is not divisible by embed dim"
        self.head_dim = head_dim
        self.scale = self.head_dim ** -0.5

        self.linear_q = torch.nn.Linear(embed_dim, num_heads * head_dim, bias=True)
        self.linear_k = torch.nn.Linear(embed_dim, num_heads * head_dim, bias=True)
        # self.linear_vq = torch.nn.Linear(embed_dim, num_heads * head_dim, bias=True)
        # self.linear_vk = torch.nn.Linear(embed_dim, num_heads * head_dim, bias=True)
        self.avg_pool = torch.nn.AdaptiveAvgPool1d(num_fields)
        self.output_layer = torch.nn.Linear(embed_dim, embed_dim, bias=True)
        
        # self.fc = torch.nn.Linear(embed_dim, 1)

    def forward(self, query, key, value, attn_mask=None, need_weights=False):
        bsz, num_fields, embed_dim = query.size()

        q = self.linear_q(query)
        q = q.transpose(0, 1).contiguous()
        q = q.view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1) # [batch size * num_heads, num_fields, head_dim]
        q = q * self.scale
        k = self.linear_k(key)
        k = k.transpose(0, 1).contiguous()
        k = k.view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        # vq = self.linear_vq(value)
        v = value.transpose(0, 1).contiguous()
        v = v.view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        attn_output_weights = torch.bmm(q, k.transpose(1, 2))

        if attn_mask is not None:
            attn_output_weights += attn_mask

        attn_output_weights = F.softmax(
            attn_output_weights, dim=-1)
        attn_output_weights = F.dropout(attn_output_weights, p=self.dropout_p, training=self.training) # [batch size * num_heads, num_fields, num_fields]
        # attn_output_weights = attn_output_weights[:, self.mask] # [bsz * num_heads, n(n-1)/2] Upper triangular matrix
        # vq and vk share size as [batch_size * num_heads, num_fields, head_dim]
        # inner_product = vq[:, self.row] * vk[:, self.col] # [bsz * num_heads, n(n-1)/2, head_dim]
        # inner_product = vq * vk

        attn_output = torch.bmm(attn_output_weights, v)
        assert list(attn_output.size()) == [bsz * self.num_heads, num_fields, self.head_dim]
        attn_output = attn_output.transpose(0, 1).contiguous().view(num_fields, bsz, embed_dim).transpose(0, 1)
        # attn_output = attn_output_weights.unsqueeze(-1) * inner_product # same shape with inner product
        # assert list(attn_output.size()) == [bsz * self.num_heads, self.num_cross_terms, self.head_dim]
        # attn_output = attn_output.transpose(0, 1).contiguous().view(self.num_cross_terms, bsz, self.embed_dim).transpose(0, 1) # [batch_size, num_cross_terms, embed_dim]
        
        attn_output = self.output_layer(attn_output)
        if need_weights:
            # average attention weights over heads
            attn_output_weights = attn_output_weights.view(bsz, self.num_heads, num_fields, num_fields)
            return attn_output, attn_output_weights.sum(dim=0) / bsz
        
        return attn_output, None


class FeaturesInteractionLayer(torch.nn.Module):
    """Encoder layer block.

    In the original paper each operation (multi-head attention or FFN) is
    postprocessed with: `dropout -> add residual -> layernorm`. In the
    tensor2tensor code they suggest that learning is more robust when
    preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *args.encoder_normalize_before* to ``True``.

    Args:
        args (argparse.Namespace): parsed command-line arguments
    """

    def __init__(self, num_fields, embed_dim, num_heads, ffn_embed_dim, dropout, activation_fn='relu', normalize_before=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ffn_embed_dim = ffn_embed_dim
        self.normalize_before = normalize_before
        self.self_attn = self.build_self_attention(num_fields, embed_dim, num_heads, dropout)
        self.self_attn_layer_norm = torch.nn.LayerNorm(embed_dim)
        self.dropout = dropout
        self.activation_fn = get_activation_fn(
            activation=activation_fn
        )
        self.activation_dropout = 0.0
        if self.activation_dropout == 0:
            # for backwards compatibility with models that use args.relu_dropout
            self.activation_dropout = self.dropout
        self.fc1 = self.build_fc1(
            embed_dim, ffn_embed_dim
        )
        self.fc2 = self.build_fc2(
            ffn_embed_dim, embed_dim
        )

        self.final_layer_norm = torch.nn.LayerNorm(embed_dim)

    def build_fc1(self, input_dim, output_dim):
        return torch.nn.Linear(input_dim, output_dim)

    def build_fc2(self, input_dim, output_dim):
        return torch.nn.Linear(input_dim, output_dim)

    def build_self_attention(self, num_fields, embed_dim, num_heads, dropout):
        return MultiheadAttentionInnerProduct(
            num_fields,
            embed_dim,
            num_heads,
            dropout=dropout
        )

    def forward(self, x, memory, attn_mask=None):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, src_len)` where padding elements are indicated by ``1``.
            attn_mask (ByteTensor): binary tensor of shape (T_tgt, T_src), where
            T_tgt is the length of query, while T_src is the length of key,
            though here both query and key is x here,
            attn_mask[t_tgt, t_src] = 1 means when calculating embedding
            for t_tgt, t_src is excluded (or masked out), =0 means it is
            included in attention

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        
        x, y = self.self_attn(
            x, memory, x, memory,
            attn_mask=attn_mask
        )
        x = F.dropout(x, p=self.dropout, training=self.training)
        # print(x.size(), y.size())
        x = residual + x
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        # residual = x
        # if self.normalize_before:
        #     x = self.final_layer_norm(x)

        # x = self.activation_fn(self.fc1(x))
        # x = F.dropout(x, p=float(self.activation_dropout), training=self.training)
        # x = self.fc2(x)
        # x = F.dropout(x, p=self.dropout, training=self.training)
        # x = residual + x
        # if not self.normalize_before:
        #     x = self.final_layer_norm(x)
        return x, y


class CrossAttentionalProductNetwork(torch.nn.Module):

    def __init__(self, num_fields, embed_dim, num_heads, ffn_embed_dim, num_layers, dropout, activation_fn='relu', normalize_before=True):
        super().__init__()
        self.layers = torch.nn.ModuleList([])
        self.layers.extend(
            [self.build_encoder_layer(num_fields, embed_dim, num_heads, ffn_embed_dim, dropout, activation_fn, normalize_before) for i in range(num_layers)]
        )
        self.dropout = dropout
        if normalize_before:
            self.layer_norm = torch.nn.LayerNorm(embed_dim)
        else:
            self.layer_norm = None
        self.fc = torch.nn.Linear(embed_dim, 1)

    def build_encoder_layer(self, num_fields, embed_dim, num_heads, ffn_embed_dim, dropout, activation_fn, normalize_before):
        return FeaturesInteractionLayer(num_fields, embed_dim, num_heads, ffn_embed_dim, dropout, activation_fn, normalize_before)

    def forward(self, x, attn_mask=None):
        # x shape: [batch_size, num_fields, embed_dim]
        x0 = x
        output = []
        for layer in self.layers:
            x, y = layer(x, x0, attn_mask)
            output.append(y)
        output = torch.cat(output, dim=1)
        if self.layer_norm is not None:
            x = self.layer_norm(x)

        return self.fc(output)


class FeaturesInteractionDecoderLayer(torch.nn.Module):
    """Encoder layer block.

    In the original paper each operation (multi-head attention or FFN) is
    postprocessed with: `dropout -> add residual -> layernorm`. In the
    tensor2tensor code they suggest that learning is more robust when
    preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *args.encoder_normalize_before* to ``True``.


    Args:
        args (argparse.Namespace): parsed command-line arguments
    """

    def __init__(self, num_fields, embed_dim, num_heads, ffn_embed_dim, dropout, activation_fn, normalize_before):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ffn_embed_dim = ffn_embed_dim
        self.normalize_before = normalize_before
        # self.self_attn = self.build_self_attention(embed_dim, num_heads, dropout)
        # self.self_attn_layer_norm = torch.nn.BatchNorm1d(num_fields)
        self.cross_attn = self.build_cross_attention(embed_dim, num_heads, dropout)
        self.cross_attn_layer_norm = torch.nn.LayerNorm(embed_dim)
        self.dropout = dropout
        self.activation_fn = get_activation_fn(
            activation=activation_fn
        )
        self.activation_dropout = 0.0
        if self.activation_dropout == 0:
            # for backwards compatibility with models that use args.relu_dropout
            self.activation_dropout = self.dropout
        self.fc1 = self.build_fc1(
            embed_dim, ffn_embed_dim
        )
        self.fc2 = self.build_fc2(
            ffn_embed_dim, embed_dim
        )

        self.final_layer_norm = torch.nn.LayerNorm(embed_dim)

    def build_fc1(self, input_dim, output_dim):
        return torch.nn.Linear(input_dim, output_dim)

    def build_fc2(self, input_dim, output_dim):
        return torch.nn.Linear(input_dim, output_dim)

    # def build_self_attention(self, embed_dim, num_heads, dropout):
    #     return torch.nn.MultiheadAttention(
    #         embed_dim,
    #         num_heads,
    #         dropout=dropout
    #     )

    def build_cross_attention(self, embed_dim, num_heads, dropout):
        return torch.nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=dropout
        )

    def forward(self, x, memory, attn_mask=None):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, src_len)` where padding elements are indicated by ``1``.
            attn_mask (ByteTensor): binary tensor of shape (T_tgt, T_src), where
            T_tgt is the length of query, while T_src is the length of key,
            though here both query and key is x here,
            attn_mask[t_tgt, t_src] = 1 means when calculating embedding
            for t_tgt, t_src is excluded (or masked out), =0 means it is
            included in attention

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        # residual = x
        # if self.normalize_before:
        #     x = self.self_attn_layer_norm(x)
        
        # x, _ = self.self_attn(
        #     x, x, x,
        #     attn_mask=attn_mask
        # )
        # x = F.dropout(x, p=self.dropout, training=self.training)
        # x = residual + x
        # if not self.normalize_before:
        #     x = self.self_attn_layer_norm(x)

        residual = x
        if self.normalize_before:
            x = self.cross_attn_layer_norm(x)
        
        x, _ = self.cross_attn(
            x, memory, memory,
            attn_mask=attn_mask
        )
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        if not self.normalize_before:
            x = self.cross_attn_layer_norm(x)

        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)

        x = self.activation_fn(self.fc1(x))
        x = F.dropout(x, p=float(self.activation_dropout), training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        if not self.normalize_before:
            x = self.final_layer_norm(x)
        return x


class CrossAttentionNetwork(torch.nn.Module):

    def __init__(self, num_fields, embed_dim, num_heads, ffn_embed_dim, num_layers, dropout, activation_fn='relu', normalize_before=True):
        super().__init__()
        self.layers = torch.nn.ModuleList([])
        self.layers.extend(
            [self.build_encoder_layer(num_fields, embed_dim, num_heads, ffn_embed_dim, dropout, activation_fn, normalize_before) for i in range(num_layers)]
        )
        self.dropout = dropout
        if normalize_before:
            self.layer_norm = torch.nn.LayerNorm(embed_dim)
        else:
            self.layer_norm = None

    def build_encoder_layer(self, num_fields, embed_dim, num_heads, ffn_embed_dim, dropout, activation_fn, normalize_before):
        return FeaturesInteractionDecoderLayer(num_fields, embed_dim, num_heads, ffn_embed_dim, dropout, activation_fn, normalize_before)

    def forward(self, x, attn_mask=None):
        # x shape: [batch_size, num_fields, embed_dim]
        x0 = x
        for layer in self.layers:
            x = layer(x, x0, attn_mask)
        if self.layer_norm is not None:
            x = self.layer_norm(x)

        return x