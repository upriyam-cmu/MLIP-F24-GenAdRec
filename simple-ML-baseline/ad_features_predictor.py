import torch
import torch.nn as nn

from typing import List, Tuple, Union, Optional


# type TupList[T] = Union[List[T], Tuple[T, ...]]


class MultiEmbedder(nn.Module):
    def __init__(
        self,
        input_cardinalities: List[int],
        embedding_dims: List[int],
        *,
        device,
    ):
        assert len(input_cardinalities) == len(embedding_dims), "input_cardinalities and embedding_dims must have same length"

        super().__init__()

        self.input_size = len(input_cardinalities)
        self.output_size = sum(embedding_dims)

        self.embedding_layers = nn.ModuleList([
            nn.Embedding(cardinality, embedding_dim, device=device) #, sparse=True)
            for cardinality, embedding_dim in zip(input_cardinalities, embedding_dims)
        ])
        assert len(self.embedding_layers) == self.input_size

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        assert len(X.shape) == 2 and X.shape[-1] == self.input_size, f"Expected input shape (*, {self.input_size}). Got {X.shape} instead."
        batch_size = X.shape[0]

        feature_split = torch.split(X, self.input_size, dim=-1)  # retains dimensionality of feature dim
        assert len(feature_split) == self.input_size

        X_embed = torch.cat([
            embedder(feature_slice.reshape(batch_size))  # returns 2D tensors
            for embedder, feature_slice in zip(self.embedding_layers, feature_split)
        ], dim=-1)

        expected_output_shape = (batch_size, self.output_size)
        assert X_embed.shape == expected_output_shape, f"{X_embed.shape} != {expected_output_shape}"

        return X_embed


class FeatureBlock(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_dims: List[int],
        output_size: int,
        device,
        *,
        activation_function: str = 'nn.ReLU()',
    ):
        super().__init__()

        hidden_layers = []
        for in_dim, out_dim in zip((input_size,)+hidden_dims[:-1], hidden_dims):
            hidden_layers.append(nn.Linear(in_dim, out_dim, device=device))
            hidden_layers.append(eval(activation_function))

        self.hidden_layers = nn.Sequential(*hidden_layers)
        self.output_layers = nn.Sequential(
            nn.Linear(hidden_dims[-1], output_size, device=device),
        )

        self.hidden_size = hidden_dims[-1]
        self.output_size = output_size

    def forward(self, input_hidden: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        new_hidden = self.hidden_layers(input_hidden)
        block_output = self.output_layers(new_hidden)

        return new_hidden, block_output


class AdFeaturesPredictor(nn.Module):
    def __init__(
        self, 
        input_cardinalities: List[int],
        embedding_dims: List[int],
        hidden_dim_specs: List[List[int]],
        output_cardinalities: List[int],
        *,
        activation_function: str = 'nn.ReLU()',
        device,
    ):
        assert len(input_cardinalities) == len(embedding_dims), "input_cardinalities and embedding_dims must have same length"
        assert len(hidden_dim_specs) == len(output_cardinalities), "hidden_dim_specs and output_cardinalities must have same length"
        
        super().__init__()

        self.embedder = MultiEmbedder(
            input_cardinalities=input_cardinalities,
            embedding_dims=embedding_dims,
            device=device,
        )

        feature_blocks, prev_output_size, input_embed_size = [], 0, self.embedder.output_size
        for hidden_dim_spec, output_size in zip(hidden_dim_specs, output_cardinalities):
            feature_block = FeatureBlock(
                input_size=input_embed_size + prev_output_size,
                hidden_dims=hidden_dim_spec,
                output_size=output_size,
                activation_function=activation_function,
                device=device,
            )
            feature_blocks.append(feature_block)
            prev_output_size = feature_block.hidden_size

        self.feature_blocks = nn.ModuleList(feature_blocks)

    def forward(self, input_features: torch.Tensor) -> List[torch.Tensor]:
        embed, hidden, outputs = self.embedder(input_features), None, []

        for feature_block in self.feature_blocks:
            hidden, output = feature_block(embed if hidden is None else torch.cat([embed, hidden], dim=-1))
            outputs.append(output)

        return tuple(outputs)
