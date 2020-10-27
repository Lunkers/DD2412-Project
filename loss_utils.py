import numpy as np
import torch.nn as nn
import torch.nn.utils as nnutils

class FlowNLL(nn.Module):
    """
    Negative log-likelihood loss as described in equation 3 in realNVP: https://arxiv.org/pdf/1605.08803.pdf
    This is the function referred to in equation 
    
    NB: we assume that p(z) is a multivariate gaussian distribution, as mentioned in the paper.
    (For qualitative experiments, they use a learnable prior, but we're only intersted in quantitative data.)
    i.e: p(z) € N(z, 0,I)

    Args:
        k: number of discrete values in each input dimension: for a normal (8-bit) image, this value is 256
    """

    def __init__(self, k=256):
        super(FlowNLL, self).__init__()
        self.k = k

    def forward(self, logp_sum, logdet, img_size, n_bins = 256):
        """
        Forward pass for the loss function
        Args:
            z: output from the model
            logdet: the log-determinant
        """
        #log-likelihood of the (assumed) gaussian for p(z)
        logdet = logdet.mean()
        pixels = np.prod(img_size[1:])
        logdet = logdet.mean()
        loss = -np.log(n_bins) * pixels
        loss = loss + logdet + logp_sum
        # prior_log_likelihood = -0.5 * (z ** 2 +np.log(2 * np.pi))
        # prior_log_likelihood = z.flatten(1).sum(-1) - np.log(self.k) * np.prod(z.size()[:1]) 
        # log_likelihood = prior_log_likelihood + logdet
        # negative_log_likelihood = -log_likelihood.mean()

        return (-loss / (np.log(2.) * pixels)).mean()

def bits_per_dimension(x, negative_log_likelihood):
    """
    Get bits per dimension for an input x

    Args:
        x: input for the model, we use this to get dimensions
        negative_log_likelihood: negative log-likelihood loss tensor
    """
    dimensions = np.prod(x.size()[1:])
    bpd = -negative_log_likelihood / (np.log(2) * dimensions)

    return bpd 
