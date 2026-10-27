"""Reusable neural network building blocks shared across ST and MM modules.

Provides factory functions for activation functions, dense (MLP) stacks,
TransformerConv-based GCN stacks, and recurrent cells.  All functions
return nn.Module or torch_geometric.nn.Sequential instances so they can
be embedded directly into any model's __init__.

Source: extracted from legacy_code/1-ST/model.py.
"""

import torch.nn as nn
from torch_geometric.nn import Sequential as PyGSequential
from torch_geometric.nn import TransformerConv


def get_activation(act_type: str = "relu") -> nn.Module:
    """Return a PyTorch activation module by name.

    Args:
        act_type: One of 'relu', 'elu', 'lelu' (leaky ReLU), 'tanh',
                  'sigmoid', 'softplus', or None (returns None).

    Returns:
        Corresponding nn.Module.

    Raises:
        ValueError: For unsupported activation names.

    Source: extracted from legacy_code/1-ST/model.py.
    """
    mapping = {
        "relu": nn.ReLU,
        "elu": nn.ELU,
        "lelu": nn.LeakyReLU,
        "tanh": nn.Tanh,
        "sigmoid": nn.Sigmoid,
        "softplus": nn.Softplus,
    }
    if act_type is None:
        return None
    if act_type not in mapping:
        raise ValueError(
            f"Unsupported activation '{act_type}'. "
            f"Choose from: {sorted(mapping)} or None."
        )
    return mapping[act_type]()


def dense_vary(
    input_dim: int,
    layer_dims: list,
    output_dim: int,
    activation: str = "relu",
    dropout: float = 0.0,
    batch_norm: bool = False,
    last_act: str = None,
) -> nn.Sequential:
    """Build a variable-depth fully-connected (MLP) stack.

    Each hidden layer is: Linear → Activation → (BatchNorm1d) → Dropout.
    The output layer is: Linear → (last_act).

    Args:
        input_dim:   Dimensionality of the input.
        layer_dims:  List of hidden layer widths.  Empty list → single
                     Linear layer from input to output.
        output_dim:  Dimensionality of the output.
        activation:  Activation applied after every hidden Linear.
        dropout:     Dropout probability applied after every hidden layer.
        batch_norm:  When True, insert BatchNorm1d after each activation.
        last_act:    Optional activation applied after the final Linear.
                     Pass None for no activation on the output.

    Returns:
        nn.Sequential containing the full stack.

    Source: extracted from legacy_code/1-ST/model.py.
    """
    layers = nn.Sequential()

    if layer_dims:
        layers.append(nn.Linear(input_dim, layer_dims[0]))
        layers.append(get_activation(activation))
        if batch_norm:
            layers.append(nn.BatchNorm1d(layer_dims[0]))
        layers.append(nn.Dropout(p=dropout))

        for i in range(len(layer_dims) - 1):
            layers.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))
            layers.append(get_activation(activation))
            if batch_norm:
                layers.append(nn.BatchNorm1d(layer_dims[i + 1]))
            layers.append(nn.Dropout(p=dropout))

        layers.append(nn.Linear(layer_dims[-1], output_dim))
    else:
        layers.append(nn.Linear(input_dim, output_dim))

    if last_act is not None:
        layers.append(get_activation(last_act))

    return layers


def GCN_vary(
    input_dim: int,
    layer_dims: list,
    output_dim: int,
    activation: str = "relu",
    last_act: str = None,
) -> PyGSequential:
    """Build a variable-depth TransformerConv GCN stack.

    Each hidden layer is: TransformerConv → Activation.
    The output layer is: TransformerConv → (last_act).

    Args:
        input_dim:  Input node-feature dimensionality.
        layer_dims: List of hidden layer widths.
        output_dim: Output node-feature dimensionality.
        activation: Activation after every hidden GCN layer.
        last_act:   Optional activation after the final GCN layer.

    Returns:
        torch_geometric.nn.Sequential with signature 'x, edge_index'.

    Source: extracted from legacy_code/1-ST/model.py.
    """
    modules = []

    if layer_dims:
        modules.append(
            (TransformerConv(input_dim, layer_dims[0]), "x, edge_index -> x")
        )
        modules.append(get_activation(activation))

        for i in range(len(layer_dims) - 1):
            modules.append(
                (TransformerConv(layer_dims[i], layer_dims[i + 1]), "x, edge_index -> x")
            )
            modules.append(get_activation(activation))

        modules.append(
            (TransformerConv(layer_dims[-1], output_dim), "x, edge_index -> x")
        )
    else:
        modules.append(
            (TransformerConv(input_dim, output_dim), "x, edge_index -> x")
        )

    if last_act is not None:
        modules.append(get_activation(last_act))

    return PyGSequential("x, edge_index", modules)


def recurrent_cell(
    input_dim: int,
    hidden_dim: int,
    rnn_type: str,
) -> nn.Module:
    """Return an LSTMCell or GRUCell.

    Args:
        input_dim:  Dimensionality of the cell input.
        hidden_dim: Dimensionality of the hidden state.
        rnn_type:   'lstm' or 'gru'.

    Returns:
        nn.LSTMCell or nn.GRUCell.

    Raises:
        ValueError: For unsupported rnn_type values.

    Source: extracted from legacy_code/1-ST/model.py.
    """
    if rnn_type == "lstm":
        return nn.LSTMCell(input_dim, hidden_dim)
    elif rnn_type == "gru":
        return nn.GRUCell(input_dim, hidden_dim)
    raise ValueError(f"Unsupported rnn_type '{rnn_type}'. Expected 'lstm' or 'gru'.")
