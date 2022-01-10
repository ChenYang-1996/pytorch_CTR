import torch
from ..layer import FeaturesEmbedding, FeaturesLinear, CrossAttentionalProductNetwork, MultiLayerPerceptron

class DeepCrossAttentionalProductNetwork(torch.nn.Module):
    def __init__(self, field_dims, embed_dim, num_heads, num_layers, mlp_dims, dropouts):
        super().__init__()
        num_fields = len(field_dims)
        self.cap = CrossAttentionalProductNetwork(num_fields, embed_dim, num_heads, num_layers, dropouts[0])
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.num_layers = num_layers
        # self.linear = FeaturesLinear(field_dims)
        self.embed_output_dim = num_fields * embed_dim
        self.attn_output_dim = num_layers * num_fields * (num_fields - 1) // 2
        # self.mlp = MultiLayerPerceptron(self.embed_output_dim, mlp_dims, dropout, output_layer=True)
        # self.linear = torch.nn.Linear(mlp_dims[-1] + num_layers * (num_fields + 1) * num_fields // 2, 1)
        self.mlp = MultiLayerPerceptron(self.attn_output_dim + self.embed_output_dim, mlp_dims, dropouts[1])
        # self._reset_parameters()

    def generate_square_subsequent_mask(self, num_fields):
        r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
        """
        mask = (torch.triu(torch.ones(num_fields, num_fields)) == 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    # def _reset_parameters(self):
    #     r"""Initiate parameters in the transformer model."""

    #     for p in self.parameters():
    #         if p.dim() > 1:
    #             torch.nn.init.xavier_uniform_(p)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        device = x.device
        attn_mask = self.generate_square_subsequent_mask(x.size(1)).to(device)
        embed_x = self.embedding(x)
        cross_term = self.cap(embed_x, attn_mask)
        # y = self.mlp_attn(cross_term.view(-1, self.attn_output_dim))
        # x = y + self.mlp(embed_x.view(-1, self.embed_output_dim))
        # y = self.mlp(embed_x.view(-1, self.embed_output_dim))
        y = torch.cat([embed_x.view(-1, self.embed_output_dim), cross_term], dim=1)
        # y = torch.cat([embed_x.view(-1, self.embed_output_dim), cross_term], dim=1)
        x = self.mlp(y)
        # y = torch.cat([cross_term, y], dim=1)
        # x = self.linear(y)
        # x = self.mlp(embed_x.view(-1, self.embed_output_dim)) + torch.sum(cross_term, dim=1, keepdim=True)
        # print(x.size())
        return torch.sigmoid(x.squeeze(1))
