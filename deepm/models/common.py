import torch
from enum import Enum
from typing import Dict
from torch import nn
import numpy as np

import torch.nn.functional as F


class LossFunction(Enum):
    """Supported loss function identifiers."""

    SHARPE = 0


def positional_encoding(max_len: int, d_model: int) -> torch.Tensor:
    """Sine-cosine positional encoding: (max_len, d_model)."""
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
    pe = torch.zeros(max_len, d_model)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


def rolling_average_conv(tensor: torch.Tensor, window_size: int, device: torch.device, feature_dim: int = 1, exponentially_weighted: bool = False) -> torch.Tensor:
    """
    Causal rolling average (or EMA) along the time dimension.

    Input:  (batch_size, sequence_length, feature_dim)
    Output: same shape
    """
    if exponentially_weighted:
        alpha = 2 / (window_size + 1)
        sequence_length = tensor.shape[1]
        ema = torch.zeros_like(tensor)
        ema[:, 0, :] = tensor[:, 0, :]
        for t in range(1, sequence_length):
            ema[:, t, :] = alpha * tensor[:, t, :] + (1 - alpha) * ema[:, t - 1, :]
        return ema

    kernel = torch.ones(1, 1, window_size, device=device) / window_size
    tensor = tensor.permute(0, 2, 1)
    tensor = F.pad(tensor, (window_size - 1, 0))
    rolling_avg = F.conv1d(tensor, kernel, groups=feature_dim)
    return rolling_avg.permute(0, 2, 1)


class DropoutNoScaling(nn.Module):
    """Dropout that does NOT rescale surviving activations."""

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x):
        if self.training:
            mask = x.new_empty(x.size()).bernoulli_(1 - self.p)
            return mask * x
        return x


class GatedLinearUnit(nn.Module):
    """Gated Linear Unit with optional dropout."""

    def __init__(self, input_size: int, hidden_size: int = None, dropout: float = None):
        super().__init__()
        if dropout is not None:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = dropout
        self.hidden_size = hidden_size or input_size
        self.fc = nn.Linear(input_size, self.hidden_size * 2)

    def forward(self, x):
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.fc(x)
        x = F.glu(x, dim=-1)
        return x


class AddNorm(nn.Module):
    """Residual addition with optional trainable gate and layer norm."""

    def __init__(self, input_size: int, trainable_add: bool = True):
        super().__init__()
        self.input_size = input_size
        self.trainable_add = trainable_add
        if self.trainable_add:
            self.mask = nn.Parameter(torch.zeros(self.input_size, dtype=torch.float))
            self.gate = nn.Sigmoid()
        self.norm = nn.LayerNorm(self.input_size)

    def forward(self, x: torch.Tensor, skip: torch.Tensor):
        if self.trainable_add:
            skip = skip * self.gate(self.mask) * 2.0
        return self.norm(x + skip)


class GateAddNorm(nn.Module):
    """GLU followed by gated residual add-and-norm."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = None,
        trainable_add: bool = False,
        dropout: float = None,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size or input_size
        self.dropout = dropout

        self.glu = GatedLinearUnit(
            self.input_size, hidden_size=self.hidden_size, dropout=self.dropout
        )
        self.add_norm = AddNorm(self.hidden_size, trainable_add=trainable_add)

    def forward(self, x, skip):
        output = self.glu(x)
        output = self.add_norm(output, skip)
        return output


class GatedResidualNetwork(nn.Module):
    """Gated Residual Network (GRN) with optional context input."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        dropout: float = 0.1,
        context_size: int = None,
        residual: bool = False,
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.context_size = context_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.residual = residual

        if self.input_size != self.output_size and not self.residual:
            residual_size = self.input_size
        else:
            residual_size = self.output_size

        if self.output_size != residual_size:
            self.resample_norm = nn.Linear(residual_size, self.output_size)

        self.fc1 = nn.Linear(self.input_size, self.hidden_size)
        self.elu = nn.ELU()

        if self.context_size is not None:
            self.context = nn.Linear(self.context_size, self.hidden_size, bias=False)

        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)

        self.gate_norm = GateAddNorm(
            input_size=self.hidden_size,
            hidden_size=self.output_size,
            dropout=self.dropout,
            trainable_add=False,
        )

    def forward(self, x, context=None, residual=None):
        if residual is None:
            residual = x

        if self.input_size != self.output_size and not self.residual:
            residual = self.resample_norm(residual)

        x = self.fc1(x)
        if context is not None:
            context = self.context(context)
            x = x + context
        x = self.elu(x)
        x = self.fc2(x)
        x = self.gate_norm(x, residual)
        return x


class VariableSelectionNetwork(nn.Module):
    """Variable Selection Network (VSN) with per-variable GRNs."""

    def __init__(
        self,
        input_sizes: Dict[str, int],
        hidden_size: int,
        dropout: float = 0.1,
        context_size: int = None,
        single_variable_grns: Dict[str, GatedResidualNetwork] = None,
        prescalers: Dict[str, nn.Linear] = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.input_sizes = input_sizes
        self.dropout = dropout
        self.context_size = context_size
        single_variable_grns = single_variable_grns or {}
        prescalers = prescalers or {}

        if self.num_inputs > 1:
            if self.context_size is not None:
                self.flattened_grn = GatedResidualNetwork(
                    self.input_size_total,
                    min(self.hidden_size, self.num_inputs),
                    self.num_inputs,
                    self.dropout,
                    self.context_size,
                    residual=False,
                )
            else:
                self.flattened_grn = GatedResidualNetwork(
                    self.input_size_total,
                    min(self.hidden_size, self.num_inputs),
                    self.num_inputs,
                    self.dropout,
                    residual=False,
                )

        self.single_variable_grns = nn.ModuleDict()
        self.prescalers = nn.ModuleDict()
        for name, input_size in self.input_sizes.items():
            if name in single_variable_grns:
                self.single_variable_grns[name] = single_variable_grns[name]
            else:
                self.single_variable_grns[name] = GatedResidualNetwork(
                    input_size,
                    min(input_size, self.hidden_size),
                    output_size=self.hidden_size,
                    dropout=self.dropout,
                )
            if name in prescalers:
                self.prescalers[name] = prescalers[name]
            else:
                self.prescalers[name] = nn.Linear(1, input_size)

        self.softmax = nn.Softmax(dim=-1)

    @property
    def input_size_total(self):
        return sum(self.input_sizes.values())

    @property
    def num_inputs(self):
        return len(self.input_sizes)

    def forward(self, x: Dict[str, torch.Tensor], context: torch.Tensor = None):
        if self.num_inputs > 1:
            var_outputs = []
            weight_inputs = []
            for name in self.input_sizes.keys():
                variable_embedding = x[name]
                if name in self.prescalers:
                    variable_embedding = self.prescalers[name](variable_embedding)
                weight_inputs.append(variable_embedding)
                var_outputs.append(self.single_variable_grns[name](variable_embedding))
            var_outputs = torch.stack(var_outputs, dim=-1)

            flat_embedding = torch.cat(weight_inputs, dim=-1)
            sparse_weights = self.flattened_grn(flat_embedding, context)
            sparse_weights = self.softmax(sparse_weights).unsqueeze(-2)

            outputs = var_outputs * sparse_weights
            outputs = outputs.sum(dim=-1)
        else:
            name = next(iter(self.single_variable_grns.keys()))
            variable_embedding = x[name]
            if name in self.prescalers:
                variable_embedding = self.prescalers[name](variable_embedding)
            outputs = self.single_variable_grns[name](variable_embedding)
            if outputs.ndim == 3:
                sparse_weights = torch.ones(
                    outputs.size(0), outputs.size(1), 1, 1, device=outputs.device
                )
            else:
                sparse_weights = torch.ones(
                    outputs.size(0), 1, 1, device=outputs.device
                )
        return outputs, sparse_weights


def causal_attention_mask(tgt_len: int, encoder_length: int = 0) -> torch.Tensor:
    """Causal attention mask for masked-fill (True = masked position)."""
    if encoder_length:
        left = torch.zeros((tgt_len, encoder_length))
        top_right = torch.ones((encoder_length, tgt_len - encoder_length))
        bottom_right = torch.triu(
            torch.full((tgt_len - encoder_length, tgt_len - encoder_length), 1),
            diagonal=1,
        )
        right = torch.cat([top_right, bottom_right])
        return torch.cat([left, right], dim=1).bool()
    else:
        return torch.triu(torch.ones(tgt_len, tgt_len), diagonal=1).bool()
