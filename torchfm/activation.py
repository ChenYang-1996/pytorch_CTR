import torch.nn.functional as F
import torch
import torch.nn as nn

class Dice(nn.Module):
     """Input shape:
        - 2 dims: [batch_size, embedding_size(features)]
        - 3 dims: [batch_size, num_features, embedding_size(features)]
    Output shape:
        - Same shape as input. """
        
    def __init__(self, emb_size, dim=2, epsilon=1e-8, device='cpu'):
        super(Dice, self).__init__()
        assert dim == 2 or dim == 3

        self.bn = nn.BatchNorm1d(emb_size, eps=epsilon)
        self.sigmoid = nn.Sigmoid()
        self.dim = dim

        # wrap alpha in nn.Parameter to make it trainable
        if self.dim == 2:
            self.alpha = nn.Parameter(torch.zeros((emb_size,)).to(device))
        else:
            self.alpha = nn.Parameter(torch.zeros((emb_size, 1)).to(device))

    def forward(self, x):
        assert x.dim() == self.dim
        if self.dim == 2:
            x_p = self.sigmoid(self.bn(x))
            out = self.alpha * (1 - x_p) * x + x_p * x
        else:
            x = torch.transpose(x, 1, 2)
            x_p = self.sigmoid(self.bn(x))
            out = self.alpha * (1 - x_p) * x + x_p * x
            out = torch.transpose(out, 1, 2)
        return out

def get_activation_fn(activation: str):
    """ Returns the activation function corresponding to `activation` """
    if activation == "relu":
        return F.relu
    # elif activation == "gelu":
    #     return gelu
    # elif activation == "gelu_fast":
    #     deprecation_warning(
    #         "--activation-fn=gelu_fast has been renamed to gelu_accurate"
    #     )
    #     return gelu_accurate
    # elif activation == "gelu_accurate":
    #     return gelu_accurate
    elif activation == "tanh":
        return torch.tanh
    elif activation == "linear":
        return lambda x: x
    elif activation == "dice":
        assert dice_dim
        return Dice(hidden_size, dice_dim)
    else:
        raise RuntimeError("--activation-fn {} not supported".format(activation))
