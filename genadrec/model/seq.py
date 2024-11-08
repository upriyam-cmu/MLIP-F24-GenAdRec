from typing import Optional, Tuple

import torch
import torch.nn as nn

from minGRU_pytorch import minGRU as minGRU_impl


class MinGRU(nn.Module):
    def __init__(self, input_size, hidden_size, batch_first=True) -> None:
        super().__init__()

        self._impl = minGRU_impl(
            dim=input_size,
            expansion_factor=(hidden_size / input_size),
        )

        self.seq_first = not batch_first

    def forward(self, X: torch.Tensor, hidden: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.seq_first:
            X = X.movedim(0, -2)  # [S, B..., D] -> [B..., S, D]

        return self._impl(X, prev_hidden=hidden, return_next_prev_hidden=True)  # y, hidden


_rnn_cell_options = {
    'GRU': nn.GRU,
    'LSTM': nn.LSTM,
    'MinGRU': MinGRU,
}


class RNN(nn.Module):
    def __init__(self, cell_type: str, input_size, hidden_size, batch_first=True) -> None:
        super().__init__()

        if cell_type not in _rnn_cell_options:
            raise ValueError(f"Invalid RNN cell type: Got {cell_type}, expected one of {tuple(_rnn_cell_options.keys())}")
        
        self.cell = _rnn_cell_options[cell_type](
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=batch_first,
        )

        self.hidden_state = None

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        y, self.hidden_state = self.cell(X, self.hidden_state)
        return y

    def reset(self) -> None:
        self.hidden_state = None
