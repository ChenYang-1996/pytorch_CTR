import numpy as np
import torch
import torch.nn.functional as F
from .activation import Dice

class FeaturesLinear(torch.nn.Module):
    #linear layer :f(x)=fc(x)+b

    def __init__(self, field_dims, output_dim=1):
        super().__init__()
        self.fc = torch.nn.Embedding(sum(field_dims), output_dim)
        self.bias = torch.nn.Parameter(torch.zeros((output_dim,)))
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.long)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        x = x + x.new_tensor(self.offsets).unsqueeze(0)
        return torch.sum(self.fc(x), dim=1) + self.bias


class FeaturesEmbedding(torch.nn.Module):
    #embedding layer, reduce the dimension

    def __init__(self, field_dims, embed_dim):
        super().__init__()
        self.embedding = torch.nn.Embedding(sum(field_dims), embed_dim)
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.long)
        torch.nn.init.xavier_uniform_(self.embedding.weight.data)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        x = x + x.new_tensor(self.offsets).unsqueeze(0)
        return self.embedding(x)


class FactorizationMachine(torch.nn.Module):
    #FM
    def __init__(self, reduce_sum=True):
        super().__init__()
        self.reduce_sum = reduce_sum

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, num_fields, embed_dim)``
        """
        square_of_sum = torch.sum(x, dim=1) ** 2
        sum_of_square = torch.sum(x ** 2, dim=1)
        ix = square_of_sum - sum_of_square
        if self.reduce_sum:
            ix = torch.sum(ix, dim=1, keepdim=True)
        return 0.5 * ix


class FieldAwareFactorizationMachine(torch.nn.Module):
    # FFM

    def __init__(self, field_dims, embed_dim):
        super().__init__()
        self.num_fields = len(field_dims)
        self.embeddings = torch.nn.ModuleList([
            torch.nn.Embedding(sum(field_dims), embed_dim) for _ in range(self.num_fields)
        ])
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.long)
        for embedding in self.embeddings:
            torch.nn.init.xavier_uniform_(embedding.weight.data)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        x = x + x.new_tensor(self.offsets).unsqueeze(0)
        xs = [self.embeddings[i](x) for i in range(self.num_fields)]
        ix = list()
        for i in range(self.num_fields - 1):
            for j in range(i + 1, self.num_fields):
                ix.append(xs[j][:, i] * xs[i][:, j])
        ix = torch.stack(ix, dim=1)
        return ix


class MultiLayerPerceptron(torch.nn.Module):
    #MLP
    def __init__(self, input_dim, embed_dims, dropout, output_layer=True, activation=False):#default activation is relu
        super().__init__()
        layers = list()
        for embed_dim in embed_dims:
            layers.append(torch.nn.Linear(input_dim, embed_dim))
            layers.append(torch.nn.BatchNorm1d(embed_dim))
            if activation:
                layers.append(Dice(embed_dim))
            else:
                layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(p=dropout))
            input_dim = embed_dim
        if output_layer:
            layers.append(torch.nn.Linear(input_dim, 1))
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, embed_dim)``
        """
        return self.mlp(x)


class InnerProductNetwork(torch.nn.Module):
    #IPNN used in PNN
    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, num_fields, embed_dim)``
        """
        num_fields = x.shape[1]
        row, col = list(), list()
        for i in range(num_fields - 1):
            for j in range(i + 1, num_fields):
                row.append(i), col.append(j)
        return torch.sum(x[:, row] * x[:, col], dim=2)


class OuterProductNetwork(torch.nn.Module):
    #OPNN used in PNN
    def __init__(self, num_fields, embed_dim, kernel_type='mat'):
        super().__init__()
        num_ix = num_fields * (num_fields - 1) // 2
        if kernel_type == 'mat':
            kernel_shape = embed_dim, num_ix, embed_dim
        elif kernel_type == 'vec':
            kernel_shape = num_ix, embed_dim
        elif kernel_type == 'num':
            kernel_shape = num_ix, 1
        else:
            raise ValueError('unknown kernel type: ' + kernel_type)
        self.kernel_type = kernel_type
        self.kernel = torch.nn.Parameter(torch.zeros(kernel_shape))
        torch.nn.init.xavier_uniform_(self.kernel.data)

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, num_fields, embed_dim)``
        """
        num_fields = x.shape[1]
        row, col = list(), list()
        for i in range(num_fields - 1):
            for j in range(i + 1, num_fields):
                row.append(i), col.append(j)
        p, q = x[:, row], x[:, col]
        if self.kernel_type == 'mat':
            kp = torch.sum(p.unsqueeze(1) * self.kernel, dim=-1).permute(0, 2, 1)
            return torch.sum(kp * q, -1)
        else:
            return torch.sum(p * q * self.kernel.unsqueeze(0), -1)


class AttentionalFactorizationMachine(torch.nn.Module):
    #AFM
    def __init__(self, embed_dim, attn_size, dropouts):
        super().__init__()
        self.attention = torch.nn.Linear(embed_dim, attn_size)
        self.projection = torch.nn.Linear(attn_size, 1)
        self.fc = torch.nn.Linear(embed_dim, 1)
        self.dropouts = dropouts

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, num_fields, embed_dim)``
        """
        num_fields = x.shape[1]
        row, col = list(), list()
        for i in range(num_fields - 1):
            for j in range(i + 1, num_fields):
                row.append(i), col.append(j)
        p, q = x[:, row], x[:, col]
        inner_product = p * q
        attn_scores = F.relu(self.attention(inner_product))
        attn_scores = F.softmax(self.projection(attn_scores), dim=1)
        attn_scores = F.dropout(attn_scores, p=self.dropouts[0], training=self.training)
        attn_output = torch.sum(attn_scores * inner_product, dim=1)
        attn_output = F.dropout(attn_output, p=self.dropouts[1], training=self.training)
        return self.fc(attn_output)


class AnovaKernel(torch.nn.Module):
    #anova kernel
    def __init__(self, order, reduce_sum=True):
        super().__init__()
        self.order = order
        self.reduce_sum = reduce_sum

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, num_fields, embed_dim)``
        """
        batch_size, num_fields, embed_dim = x.shape
        a_prev = torch.ones((batch_size, num_fields + 1, embed_dim), dtype=torch.float).to(x.device)
        for t in range(self.order):
            a = torch.zeros((batch_size, num_fields + 1, embed_dim), dtype=torch.float).to(x.device)
            a[:, t+1:, :] += x[:, t:, :] * a_prev[:, t:-1, :]
            a = torch.cumsum(a, dim=1)
            a_prev = a
        if self.reduce_sum:
            return torch.sum(a[:, -1, :], dim=-1, keepdim=True)
        else:
            return a[:, -1, :]


class CrossNetwork(torch.nn.Module):
    #CrossNetwork used in DCN
    def __init__(self, input_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        self.w = torch.nn.ModuleList([
            torch.nn.Linear(input_dim, 1, bias=False) for _ in range(num_layers)
        ])
        self.b = torch.nn.ParameterList([
            torch.nn.Parameter(torch.zeros((input_dim,))) for _ in range(num_layers)
        ])

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, num_fields, embed_dim)``
        """
        x0 = x
        for i in range(self.num_layers):
            xw = self.w[i](x)
            x = x0 * xw + self.b[i] + x
        return x

class CrossProductNetwork(torch.nn.Module):
    #CrossProductNetwork used in DCN
    def __init__(self, num_fields, embed_dim, num_heads, dropout=0.2, kernel_type='mat'):
        super().__init__()
        num_ix = num_fields * (num_fields - 1) // 2
        if kernel_type == 'mat':
            kernel_shape = embed_dim, num_ix, embed_dim
        elif kernel_type == 'vec':
            kernel_shape = num_ix, embed_dim
        elif kernel_type == 'num':
            kernel_shape = num_ix, 1
        else:
            raise ValueError('unknown kernel type: ' + kernel_type)
        self.kernel_type = kernel_type
        self.kernel = torch.nn.Parameter(torch.zeros(kernel_shape))
        self.avg_pool = torch.nn.AdaptiveAvgPool1d(num_fields)
        # self.fc = torch.nn.Linear(embed_dim, 1)
        self.attn = MultiheadAttentionInnerProduct(num_fields, embed_dim, num_heads, dropout)
        torch.nn.init.xavier_uniform_(self.kernel.data)

    def forward(self, x, x0, attn_mask=None):
        """
        :param x: Float tensor of size ``(batch_size, num_fields, embed_dim)``
        """
        bsz, num_fields, embed_dim = x0.size()
        row, col = list(), list()
        for i in range(num_fields - 1):
            for j in range(i + 1, num_fields):
                row.append(i), col.append(j)

        if self.kernel_type == 'mat':
            x, _ = self.attn(x, x, x, attn_mask)
            p, q = x[:, row], x0[:, col]
            kp = torch.sum(p.unsqueeze(1) * self.kernel, dim=-1).permute(0, 2, 1)  # (bsz, n(n-1)/2, embed_dim)
            kpq = kp * q  # Outer product

            x = self.avg_pool(kpq.permute(0, 2, 1)).permute(0, 2, 1)  # (bsz, n, embed_dim)

            return x, torch.sum(kpq, dim=-1)
        else:
            p, q = x[:, row], x0[:, col]# added by yc
            return torch.sum(p * q * self.kernel.unsqueeze(0), -1)


class CompressedInteractionNetwork(torch.nn.Module):
    #CIN used in XDeepFM to replace the Crossnetwork in DCN
    def __init__(self, input_dim, cross_layer_sizes, split_half=True):
        super().__init__()
        self.num_layers = len(cross_layer_sizes)
        self.split_half = split_half
        self.conv_layers = torch.nn.ModuleList()
        prev_dim, fc_input_dim = input_dim, 0
        for i in range(self.num_layers):
            cross_layer_size = cross_layer_sizes[i]
            self.conv_layers.append(torch.nn.Conv1d(input_dim * prev_dim, cross_layer_size, 1,
                                                    stride=1, dilation=1, bias=True))
            if self.split_half and i != self.num_layers - 1:
                cross_layer_size //= 2
            prev_dim = cross_layer_size
            fc_input_dim += prev_dim
        self.fc = torch.nn.Linear(fc_input_dim, 1)

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, num_fields, embed_dim)``
        """
        xs = list()
        x0, h = x.unsqueeze(2), x
        for i in range(self.num_layers):
            x = x0 * h.unsqueeze(1)
            batch_size, f0_dim, fin_dim, embed_dim = x.shape
            x = x.view(batch_size, f0_dim * fin_dim, embed_dim)
            x = F.relu(self.conv_layers[i](x))
            if self.split_half and i != self.num_layers - 1:
                x, h = torch.split(x, x.shape[1] // 2, dim=1)
            else:
                h = x
            xs.append(x)
        return self.fc(torch.sum(torch.cat(xs, dim=1), 2))
###########################################################
# class CrossAttentionalProductNetwork(torch.nn.Module):
#     #CAP used in DCAP
#
#     def __init__(self, num_fields, embed_dim, num_heads, num_layers, dropout, kernel_type='mat'):
#         super().__init__()
#         self.layers = torch.nn.ModuleList([])
#         self.layers.extend(
#             [self.build_encoder_layer(num_fields=num_fields, embed_dim=embed_dim, num_heads=num_heads,
#                                     dropout=dropout, kernel_type=kernel_type) for _ in range(num_layers)]
#         )
#         # self.layers = torch.nn.ModuleList([
#         #     torch.nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout) for _ in range(num_layers)
#         # ])
#         # self.norm = torch.nn.BatchNorm1d(num_fields * (num_fields - 1) // 2)
#         # self.avg_pool = torch.nn.AdaptiveAvgPool1d(num_fields)
#         # self.fc = torch.nn.Linear(embed_dim, 1)
#
#     def build_encoder_layer(self, num_fields, embed_dim, num_heads, dropout, kernel_type='mat'):
#         return CrossProductNetwork(num_fields=num_fields, embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, kernel_type=kernel_type)
#
#     def forward(self, x, attn_mask=None):
#         x0 = x
#         output = []
#         for layer in self.layers:
#             x, y = layer(x, x0, attn_mask)
#             output.append(y)
#         output = torch.cat(output, dim=1)
#
#         return output
#############################
#####DACP

from .activation import get_activation_fn

######要将layer.py和attention_layer.py合并

class MultiheadAttentionInnerProduct(torch.nn.Module):
    '''
    Used in DCAP for self attention
    '''

    def __init__(self, num_fields, embed_dim, num_heads, dropout):
        super().__init__()
        self.num_fields = num_fields
        self.mask = (torch.triu(torch.ones(num_fields, num_fields), diagonal=1) == 1) #Returns the upper triangular part of the tensor
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

    def forward(self, query, key, value, attn_mask=None, need_weights=False): #the default:attn_mask=None
        bsz, num_fields, embed_dim = query.size()

        q = self.linear_q(query)
        q = q.transpose(0, 1).contiguous() #deep copy
        q = q.view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)  # [batch size * num_heads, num_fields, head_dim]
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

        attn_output_weights = F.softmax(attn_output_weights, dim=-1)
        attn_output_weights = F.dropout(attn_output_weights, p=self.dropout_p,
                                        training=self.training)  # [batch size * num_heads, num_fields, num_fields]
        attn_output = torch.bmm(attn_output_weights, v)

        assert list(attn_output.size()) == [bsz * self.num_heads, num_fields, self.head_dim]
        attn_output = attn_output.transpose(0, 1).contiguous().view(num_fields, bsz, embed_dim).transpose(0, 1)

        attn_output = self.output_layer(attn_output)
        if need_weights:
            # average attention weights over heads
            attn_output_weights = attn_output_weights.view(bsz, self.num_heads, num_fields, num_fields)
            return attn_output, attn_output_weights.sum(dim=0) / bsz

        return attn_output, None


# class CrossAttentionalProductNetwork(torch.nn.Module):
#     '''
#     used in DCAP for cross attention block
#     '''
#
#     def __init__(self, num_fields, embed_dim, num_heads, ffn_embed_dim, num_layers, dropout, activation_fn='relu',
#                  normalize_before=True):
#         super().__init__()
#         self.layers = torch.nn.ModuleList([])
#         self.layers.extend(
#             [self.build_encoder_layer(num_fields, embed_dim, num_heads, ffn_embed_dim, dropout, activation_fn,
#                                       normalize_before) for i in range(num_layers)]#num_layers
#         )
#         self.dropout = dropout
#         if normalize_before:
#             self.layer_norm = torch.nn.LayerNorm(embed_dim)
#         else:
#             self.layer_norm = None
#         self.fc = torch.nn.Linear(embed_dim, 1)
#
#     def build_encoder_layer(self, num_fields, embed_dim, num_heads, ffn_embed_dim, dropout, activation_fn,
#                             normalize_before):
#         return FeaturesInteractionLayer(num_fields, embed_dim, num_heads, ffn_embed_dim, dropout, activation_fn,
#                                         normalize_before)
#
#     def forward(self, x, attn_mask=None):
#         # x shape: [batch_size, num_fields, embed_dim]
#         x0 = x
#         output = []
#         for layer in self.layers:
#             x, y = layer(x, x0, attn_mask)
#             output.append(y)
#         output = torch.cat(output, dim=1)
#         if self.layer_norm is not None:
#             x = self.layer_norm(x)
#
#         return self.fc(output)

class CrossAttentionalProductNetwork(torch.nn.Module):

    def __init__(self, num_fields, embed_dim, num_heads, num_layers, dropout, kernel_type='mat'):
        super().__init__()
        self.layers = torch.nn.ModuleList([])
        self.layers.extend(
            [self.build_encoder_layer(num_fields=num_fields, embed_dim=embed_dim, num_heads=num_heads,
                                    dropout=dropout, kernel_type=kernel_type) for _ in range(num_layers)]
        )
        # self.layers = torch.nn.ModuleList([
        #     torch.nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout) for _ in range(num_layers)
        # ])
        # self.norm = torch.nn.BatchNorm1d(num_fields * (num_fields - 1) // 2)
        # self.avg_pool = torch.nn.AdaptiveAvgPool1d(num_fields)
        # self.fc = torch.nn.Linear(embed_dim, 1)

    def build_encoder_layer(self, num_fields, embed_dim, num_heads, dropout, kernel_type='mat'):
        return CrossProductNetwork(num_fields=num_fields, embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, kernel_type=kernel_type)

    def forward(self, x, attn_mask=None):
        x0 = x
        output = []
        for layer in self.layers:
            x, y = layer(x, x0, attn_mask)
            output.append(y)
        output = torch.cat(output, dim=1)

        return output
#######################################
class FeaturesInteractionLayer(torch.nn.Module):
    """Used in DCAP for Encoder layer block.
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

    def __init__(self, num_fields, embed_dim, num_heads, ffn_embed_dim, dropout, activation_fn='relu',
                 normalize_before=True):
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
        # self.fc1 = self.build_fc1(
        #     embed_dim, ffn_embed_dim
        # )
        # self.fc2 = self.build_fc2(
        #     ffn_embed_dim, embed_dim
        # )

        self.final_layer_norm = torch.nn.LayerNorm(embed_dim)
    #
    # def build_fc1(self, input_dim, output_dim):
    #     return torch.nn.Linear(input_dim, output_dim)
    #
    # def build_fc2(self, input_dim, output_dim):
    #     return torch.nn.Linear(input_dim, output_dim)

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
            attn_mask ###########debug
        )
        x = F.dropout(x, p=self.dropout, training=self.training)
        # print(x.size(), y.size())
        x = residual + x
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        return x, y



class FeaturesInteractionDecoderLayer(torch.nn.Module):
    """Used in DCAP for decoder layer block.

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

    def __init__(self, num_fields, embed_dim, num_heads, ffn_embed_dim, num_layers, dropout, activation_fn='relu',
                 normalize_before=True):
        super().__init__()
        self.layers = torch.nn.ModuleList([])
        self.layers.extend(
            [self.build_encoder_layer(num_fields, embed_dim, num_heads, ffn_embed_dim, dropout, activation_fn,
                                      normalize_before) for i in range(num_layers)]
        )
        self.dropout = dropout
        if normalize_before:
            self.layer_norm = torch.nn.LayerNorm(embed_dim)
        else:
            self.layer_norm = None

    def build_encoder_layer(self, num_fields, embed_dim, num_heads, ffn_embed_dim, dropout, activation_fn,
                            normalize_before):
        return FeaturesInteractionDecoderLayer(num_fields, embed_dim, num_heads, ffn_embed_dim, dropout, activation_fn,
                                               normalize_before)

    def forward(self, x, attn_mask=None):
        # x shape: [batch_size, num_fields, embed_dim]
        x0 = x
        for layer in self.layers:
            x = layer(x, x0, attn_mask)
        if self.layer_norm is not None:
            x = self.layer_norm(x)

        return x

#############################################
#############################################
class Activation_Unit(torch.nn.Module):
    """
    Activation_Unit used by DIN model.

    Reference:
        Deep Interest Network for Click-Through Rate Prediction
    Activation_Unit layer.
    Parameters
    ----------
    input_size : int
        Size of input.
    hidden_layers : iterable
        Hidden layer sizes.
    dropout : float
        Dropout rate.
    activation : str
        Name of activation function. relu, prelu and sigmoid are supported.
    return_scores : bool
        Return Activation_Unit scores instead of weighted sum pooling result.
    """

    def __init__(
            self,
            input_size,
            hidden_layers,
            dropout=0.0,
            return_scores=False):
        super(Activation_Unit, self).__init__()
        self.return_scores = return_scores
        self.mlp = MultiLayerPerceptron(
            input_dim=input_size,  # input_size * 4
            embed_dims=hidden_layers,
            dropout=dropout,
            activation=True,  # use dice as activation
            output_layer=False)
        self.fc = torch.nn.Linear(hidden_layers[-1], 1)

    def forward(self, query, keys, keys_length):
        """
        Parameters
        ----------
        query: 2D tensor, [B, H]
        keys: 3D tensor, [B, T, H]
        keys_length: 1D tensor, [B]
        Returns
        -------
        outputs: 2D tensor, if return_scores=False [B, H], otherwise [B, T]
        """
        batch_size, max_length, dim = keys.size()  # [B, T, H]

        query = query.unsqueeze(1).expand(-1, max_length, -1)

        din_all = torch.cat(
            [query, keys, query - keys, query * keys], dim=-1)

        din_all = din_all.view(batch_size * max_length, -1)

        outputs = self.mlp(din_all)

        outputs = self.fc(outputs).view(batch_size, max_length)  # [B, T]

        # Scale
        outputs = outputs / (dim ** 0.5)

        # Mask
        mask = (torch.arange(max_length, device=keys_length.device).repeat(
            batch_size, 1) < keys_length.view(-1, 1))
        outputs[~mask] = -np.inf

        # Activation
        outputs = F.softmax(outputs, dim=1)  # [B, T]

        if not self.return_scores:
            # Weighted sum
            outputs = torch.matmul(
                outputs.unsqueeze(1), keys).squeeze()  # [B, H]

        return outputs

#################
'''DCNV2'''


class CrossNetMatrix(torch.nn.Module):
    """
        CrossNet of DCN-v2
    """
    def __init__(self, in_features, layer_num=2):
        super(CrossNetMatrix, self).__init__()
        self.layer_num = layer_num
        # Cross中的W参数 (layer_num,  [W])
        self.weights = torch.nn.Parameter(torch.Tensor(self.layer_num, in_features, in_features))
        # Cross中的b参数 (layer_num, [B])
        self.bias = torch.nn.Parameter(torch.Tensor(self.layer_num, in_features, 1))

        # Init
        for i in range(self.layer_num):
            torch.nn.init.xavier_normal_(self.weights[i])
        for i in range(self.layer_num):
            torch.nn.init.zeros_(self.bias[i])

    def forward(self, x):
        """
            x : batch_size  *  in_features
        """
        x0 = x.unsqueeze(2)
        xl = x.unsqueeze(2)
        for i in range(self.layer_num):
            tmp = torch.matmul(self.weights[i], xl) + self.bias[i]
            xl = x0 * tmp + xl
        xl = xl.squeeze(2)

        return xl


class CrossNetMix(torch.nn.Module):
    """
        CrossNet of DCN-V2 with Mixture of Low-rank Experts
        公式如下：
            G_i(xl) = Linear(xl)
            E_i(xl) = x0·(Ul*g(Cl*g(Vl*xl)) + bl)
            g() = tanh activate func
    """

    def __init__(self, in_features, low_rank=16, expert_num=4, layer_num=2):
        super(CrossNetMix, self).__init__()
        self.layer_num = layer_num
        self.expert_num = expert_num

        # Cross中的U参数(layer_num, expert_num, in_features, low_rank)
        self.U_params = torch.nn.Parameter(torch.Tensor(layer_num, expert_num, in_features, low_rank))
        # Cross中的V^T参数(layer_num, expert_num, low_rank, in_features)
        self.V_params = torch.nn.Parameter(torch.Tensor(layer_num, expert_num, low_rank, in_features))
        # Cross中的C参数(layer_num, expert_num, low_rank, low_rank)
        self.C_params = torch.nn.Parameter(torch.Tensor(layer_num, expert_num, low_rank, low_rank))
        # Cross中的bias(layer_num, in_features, 1)
        self.bias = torch.nn.Parameter(torch.Tensor(layer_num, in_features, 1))

        # MOE 中的门控gate
        self.gates = torch.nn.ModuleList([torch.nn.Linear(in_features, 1, bias=False) for i in range(expert_num)])

        # Init
        for i in range(self.layer_num):
            torch.nn.init.xavier_normal_(self.U_params[i])
            torch.nn.init.xavier_normal_(self.V_params[i])
            torch.nn.init.xavier_normal_(self.C_params[i])
        for i in range(self.layer_num):
            torch.nn.init.zeros_(self.bias[i])

    def forward(self, x):
        """
            x : batch_size  *  in_features
        """
        x0 = x.unsqueeze(2)
        xl = x.unsqueeze(2)
        for i in range(self.layer_num):
            expert_outputs = []
            gate_scores = []
            for expert in range(self.expert_num):
                # gate score : G(xl)
                gate_scores.append(self.gates[expert](xl.squeeze(2)))

                # cross part
                # g(Vl·xl))
                tmp = torch.tanh(torch.matmul(self.V_params[i][expert], xl))
                # g(Cl·g(Vl·xl))
                tmp = torch.tanh(torch.matmul(self.C_params[i][expert], tmp))
                # Ul·g(Cl·g(Vl·xl)) + bl
                tmp = torch.matmul(self.U_params[i][expert], tmp) + self.bias[i]
                # E_i(xl) = x0·(Ul·g(Cl·g(Vl·xl)) + bl)
                tmp = x0 * tmp
                expert_outputs.append(tmp.squeeze(2))

            expert_outputs = torch.stack(expert_outputs, 2)  # batch * in_features * expert_num
            gate_scores = torch.stack(gate_scores, 1)  # batch * expert_num * 1
            MOE_out = torch.matmul(expert_outputs, gate_scores.softmax(1))
            xl = MOE_out + xl  # batch * in_features * 1

        xl = xl.squeeze(2)

        return xl


class DeepCrossNetv2(torch.nn.Module):
    """
        Deep Cross Network V2
    """

    def __init__(self, feature_fields, embed_dim, layer_num, mlp_dims, dropout=0.1,
                 cross_method='Mix', model_method='parallel'):
        """
        """
        super(DeepCrossNetv2, self).__init__()
        self.feature_fields = feature_fields
        self.offsets = np.array((0, *np.cumsum(feature_fields)[:-1]), dtype=np.long)
        self.model_method = model_method

        # Embedding layer
        self.embedding = torch.nn.Embedding(sum(feature_fields) + 1, embed_dim)
        torch.torch.nn.init.xavier_uniform_(self.embedding.weight.data)
        self.embedding_out_dim = len(feature_fields) * embed_dim

        # DNN layer
        dnn_layers = []
        input_dim = self.embedding_out_dim
        self.mlp_dims = mlp_dims
        for mlp_dim in mlp_dims:
            # 全连接层
            dnn_layers.append(torch.nn.Linear(input_dim, mlp_dim))
            dnn_layers.append(torch.nn.BatchNorm1d(mlp_dim))
            dnn_layers.append(torch.nn.ReLU())
            dnn_layers.append(torch.nn.Dropout(p=dropout))
            input_dim = mlp_dim
        self.mlp = torch.nn.Sequential(*dnn_layers)

        if cross_method == 'Mix':
            self.CrossNet = CrossNetMix(in_features=self.embedding_out_dim)
        elif cross_method == 'Matrix':
            self.CrossNet = CrossNetMatrix(in_features=self.embedding_out_dim)
        else:
            raise NotImplementedError

        # predict layer
        if self.model_method == 'parallel':
            self.fc = torch.nn.Linear(self.mlp_dims[-1] + self.embedding_out_dim, 1)
        elif self.model_method == 'stack':
            self.fc = torch.nn.Linear(self.mlp_dims[-1], 1)
        else:
            raise NotImplementedError

    def forward(self, x):
        tmp = x + x.new_tensor(self.offsets).unsqueeze(0)

        # embeded dense vector
        embeded_x = self.embedding(tmp).view(-1, self.embedding_out_dim)
        if self.model_method == 'parallel':
            # Dtorch.nn out
            mlp_part = self.mlp(embeded_x)
            # Cross part
            cross = self.CrossNet(embeded_x)
            # stack output
            out = torch.cat([cross, mlp_part], dim=1)
        elif self.model_method == 'stack':
            # Cross part
            cross = self.CrossNet(embeded_x)
            # Dtorch.nn out
            out = self.mlp(cross)
        # predict out
        out = self.fc(out)
        out = torch.sigmoid(out.squeeze(1))

        return out
