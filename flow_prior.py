import torch
import torch.nn as nn
import torch.distributions as td
from flow import Flow, GaussianBase, MaskedCouplingLayer

class FlowDistribution(td.Distribution):
    """
    A thin wrapper that turns the Flow model into a torch.distributions.Distribution
    so we can call sample() and log_prob() in a standard way.
    """
    def __init__(self, flow_model):
        # You can turn off validation to avoid overhead in checking shapes.
        super().__init__(validate_args=False)
        self.flow_model = flow_model

    def sample(self, sample_shape=torch.Size()):
        """
        Draw samples from the flow-based distribution.
        """
        # The Flow object’s .sample(...) already does base.sample(...), then transforms it.
        return self.flow_model.sample(sample_shape)

    def log_prob(self, value):
        """
        Compute log p(value) under the flow-based distribution.
        """
        return self.flow_model.log_prob(value)


class FlowPrior(nn.Module):
    """
    Wraps a normalizing flow (with a Gaussian base) as a prior distribution for a VAE.
    """
    def __init__(self, M, num_transforms=4, hidden_dim=64):
        """
        Parameters:
        -----------
        M : int
            Dimensionality of the latent space.
        num_transforms : int
            Number of affine coupling layers in the flow.
        hidden_dim : int
            Width of the hidden layers for the scale/translation networks.
        """
        super().__init__()
        self.M = M

        # 1) Define the base distribution (standard Gaussian in dimension M).
        base = GaussianBase(M)

        # 2) Construct a sequence of MaskedCouplingLayers.
        transformations = []
        
        # Simple "alternating" mask: half channels are 1, half are 0.
        # (You can modify this to checkboard or random if you prefer.)
        mask = torch.tensor([1 if i % 2 == 0 else 0 for i in range(M)]).float()
        
        for i in range(num_transforms):
            # For each coupling layer, define scale_net and translation_net:
            scale_net = nn.Sequential(
                nn.Linear(M, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, M),
                nn.Tanh()
            )
            translation_net = nn.Sequential(
                nn.Linear(M, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, M),
                nn.Tanh()
            )
            
            transformations.append(MaskedCouplingLayer(scale_net, translation_net, mask))
            
            # Flip the mask for the next coupling layer
            mask = 1 - mask
        
        # 3) Put it all together into a Flow.
        self.flow = Flow(base, transformations)

    def forward(self):
        """
        Return a Distribution object whose .sample() and .log_prob() call
        the flow's sample/log_prob under the learned flow prior.
        """
        return FlowDistribution(self.flow)
