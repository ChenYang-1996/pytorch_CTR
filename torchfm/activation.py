import torch.nn.functional as F
import torch

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
    else:
        raise RuntimeError("--activation-fn {} not supported".format(activation))