import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaptiveResonanceLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(AdaptiveResonanceLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Define weights and biases as trainable parameters
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))

        # Initialize familiarity_index as a regular tensor, not as nn.Parameter
        self.familiarity_index = torch.ones(out_features)

        # Initialize parameters
        nn.init.uniform_(self.weight, -0.1, 0.1)
        nn.init.zeros_(self.bias)

    def forward(self, input):
        # Basic linear transformation
        output = F.linear(input, self.weight, self.bias)

        # Calculate pattern familiarity
        pattern_familiarity = torch.matmul(input, self.weight.t())

        # Update familiarity index
        self.familiarity_index = self.familiarity_index * 0.99 + pattern_familiarity.mean(dim=0) * 0.01
        self.familiarity_index = self.familiarity_index.detach()  # Detach from the computation graph

        # Dynamic weight adjustment based on familiarity
        adjusted_weights = self.weight * (1 / (self.familiarity_index.unsqueeze(1) + 1e-6))

        # Final output with adjusted weights
        output = F.linear(input, adjusted_weights, self.bias)

        return output