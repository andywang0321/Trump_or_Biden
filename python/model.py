import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class FCNet(nn.Module):
    def __init__(
        self, 
        input_size: int, 
        hidden_sizes: list[int],
        input_dropout: float,
        hidden_dropout: float,
        output_size: int = 1
    ) -> None:
        super().__init__()

        layers_ = [input_size] + hidden_sizes
        _layers = hidden_sizes + [output_size]

        self.num_params = np.sum(np.array(layers_) * np.array(_layers) + np.array(_layers))
        self.num_layers = len(hidden_sizes) + 1
        self.name = f'{self.num_layers}-Layer FC-Net ({self.num_params} parameters, dropout = {hidden_dropout})'

        layers = zip(
            [input_dropout] + [hidden_dropout for _ in hidden_sizes],
            layers_,
            _layers
        )

        for layer_num, (dropout, dim_in, dim_out) in enumerate(layers):
            setattr(
                self,
                f'fc_{layer_num}',
                nn.Linear(dim_in, dim_out)
            )
            setattr(
                self,
                f'batchnorm_{layer_num}',
                nn.BatchNorm1d(dim_out)
            )
            setattr(
                self,
                f'dropout_{layer_num}',
                nn.Dropout(dropout)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        for layer_num in range(self.num_layers):
            fc = getattr(
                self,
                f'fc_{layer_num}'
            )
            dropout = getattr(
                self,
                f'dropout_{layer_num}',
            )
            batchnorm = getattr(
                self, 
                f'batchnorm_{layer_num}'
            )
            x = fc(x)
            x = batchnorm(x)
            if layer_num + 1 < self.num_layers:
                x = F.relu(x)
            x = dropout(x)
        
        return F.sigmoid(x)