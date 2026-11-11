import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    """Stable original negative-only objective, with optional standard InfoNCE."""
    def __init__(self, batch_size=None, temperature=1.0, device=None, mode='legacy'):
        super().__init__()
        if temperature <= 0 or mode not in ('legacy', 'infonce'):
            raise ValueError('Invalid contrastive settings')
        self.temperature, self.mode = temperature, mode

    def forward(self, h_i, h_j, weight=None):
        if h_i.shape != h_j.shape or h_i.shape[0] < 2:
            raise ValueError('Contrastive loss needs matching shapes and at least two samples')
        logits = h_i @ h_j.T / self.temperature
        if self.mode == 'infonce':
            loss = F.cross_entropy(logits, torch.arange(len(h_i), device=h_i.device))
        else:
            negatives = logits.masked_fill(torch.eye(len(h_i), device=h_i.device,
                                                       dtype=torch.bool), float('-inf'))
            loss = (torch.logsumexp(negatives, dim=1) - logits.diagonal()).mean()
        return loss if weight is None else weight * loss


def training_objective(model, xs, contrastive, alpha=1.0, beta=1.0, gamma=1.0,
                       heldout_view=None):
    xrs, zs, rs, H, new_rs, aux = model(xs, return_aux=True,
                                      heldout_view=heldout_view)
    rec = torch.stack([F.mse_loss(xr, x) for xr, x in zip(xrs, xs)]).mean()
    fcl = sum(contrastive(H, r, aux['pair_weights'][p])
              for p, r in enumerate(new_rs))
    # No single-view cl loss; no outer proximal loss.
    terms = {'rec': rec, 'fcl': fcl, 'cv': aux['cv'], 'fit': aux['fit']}
    total = rec + alpha * fcl + beta * terms['cv'] + gamma * terms['fit']
    return total, terms
