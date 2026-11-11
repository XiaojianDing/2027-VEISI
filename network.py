import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import normalize


# Encoder
class Encoder(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, 2000),
            nn.ReLU(),
            nn.Linear(2000, feature_dim),
        )

    def forward(self, x):
        return self.encoder(x)


# Decoder
class Decoder(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(feature_dim, 2000),
            nn.ReLU(),
            nn.Linear(2000, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, input_dim)
        )

    def forward(self, x):
        return self.decoder(x)


class IterativeSemanticFusion(nn.Module):
    """Unrolled robust latent inference; no pair matching or recursive averaging."""
    def __init__(self, views, feature_dim, semantic_dim, steps=3,
                 step_size=0.5, proximal=0.1, robust_c=1.0, hidden_dim=128):
        super().__init__()
        if views < 2 or steps < 0 or min(feature_dim, semantic_dim, hidden_dim) < 1:
            raise ValueError("Need at least two views and nonnegative inference steps")
        if step_size <= 0 or proximal < 0 or robust_c <= 0:
            raise ValueError("Invalid inference hyperparameters")
        self.views, self.feature_dim = views, feature_dim
        self.steps, self.step_size = steps, step_size
        self.proximal, self.robust_c = proximal, robust_c
        self.initializer = nn.Sequential(nn.Linear(feature_dim, 256), nn.ReLU(),
                                         nn.Linear(256, semantic_dim))
        # Tanh is smooth, supporting differentiation through energy gradients.
        self.predictors = nn.ModuleList([
            nn.Sequential(nn.Linear(semantic_dim, hidden_dim), nn.Tanh(),
                          nn.Linear(hidden_dim, feature_dim)) for _ in range(views)])

    def targets(self, zs):
        # RMS-scale normalized codes: squared error / d stays O(1), not O(1/d).
        return [F.normalize(z, dim=1) * (self.feature_dim ** 0.5) for z in zs]

    def robust_error(self, prediction, target):
        squared = (prediction - target.detach()).square().mean(dim=1)
        c2 = self.robust_c ** 2
        return c2 * torch.log1p(squared / c2)

    def energy(self, h, h0, targets, observed):
        errors = [self.robust_error(self.predictors[v](h), targets[v])
                  for v in observed]
        fit = torch.stack(errors, dim=1).mean(dim=1)
        anchor = 0.5 * self.proximal * (h - h0).square().sum(dim=1)
        return fit + anchor

    def forward(self, zs, observed=None):
        if len(zs) != self.views:
            raise ValueError("Unexpected number of views")
        observed = tuple(range(self.views)) if observed is None else tuple(observed)
        if not observed or len(set(observed)) != len(observed):
            raise ValueError("Observed view indices must be nonempty and unique")
        if any(v < 0 or v >= self.views for v in observed):
            raise ValueError("Observed view index out of range")
        differentiable = self.training and torch.is_grad_enabled()
        # Evaluation still needs local first derivatives, including under no_grad.
        with torch.enable_grad():
            targets = self.targets(zs)
            mean_observed = torch.stack([targets[v] for v in observed]).mean(dim=0)
            h0 = self.initializer(mean_observed)
            if not differentiable:
                h0 = h0.detach()
            h = h0 if differentiable else h0.clone().requires_grad_(True)
            fixed_targets = [target.detach() for target in targets]
            for _ in range(self.steps):
                energy = self.energy(h, h0, fixed_targets, observed)
                # Sum over samples: each update is independent of batch size.
                gradient, = torch.autograd.grad(energy.sum(), h,
                                                create_graph=differentiable)
                h = h - self.step_size * gradient
                if not differentiable:
                    h = h.detach().requires_grad_(True)
            return h if differentiable else h.detach()

    def fit_loss(self, h, zs, predicted=None):
        predicted = tuple(range(self.views)) if predicted is None else tuple(predicted)
        targets = self.targets(zs)
        return torch.stack([self.robust_error(self.predictors[v](h), targets[v]).mean()
                            for v in predicted]).mean()


class GlobalGuidedAttention(nn.Module):
    """Persistent, trainable counterpart of the original batch-created MHA."""
    def __init__(self, semantic_dim, num_heads=4):
        super().__init__()
        if semantic_dim % num_heads:
            raise ValueError("high_feature_dim must be divisible by num_heads")
        self.query = nn.Linear(semantic_dim, semantic_dim, bias=False)
        self.attention = nn.MultiheadAttention(semantic_dim, num_heads,
                                               batch_first=True, dropout=0.0)

    def forward(self, rs, H):
        keys = torch.stack(rs, dim=1)
        _, weights = self.attention(self.query(H).unsqueeze(1), keys, keys)
        # Retain original aggregation and second softmax, now with gradients.
        return F.softmax(weights.mean(dim=(0, 1)), dim=0)


class Network(nn.Module):
    def __init__(self, view, input_size, feature_dim, high_feature_dim, device,
                 lambda_score=0.5, inference_steps=3, inference_lr=0.5,
                 proximal_lambda=0.1, robust_c=1.0, predictor_hidden=128,
                 num_heads=4):
        super().__init__()
        self.view, self.device = view, device
        # lambda_score accepted only for old constructor compatibility.
        self.encoders = nn.ModuleList([Encoder(input_size[v], feature_dim)
                                       for v in range(view)])
        self.decoders = nn.ModuleList([Decoder(input_size[v], feature_dim)
                                       for v in range(view)])
        self.common_information_module = nn.Linear(feature_dim, high_feature_dim)
        self.semantic_fusion = IterativeSemanticFusion(
            view, feature_dim, high_feature_dim, inference_steps, inference_lr,
            proximal_lambda, robust_c, predictor_hidden)
        self.view_attention = GlobalGuidedAttention(high_feature_dim, num_heads)
        self.to(device)

    def encode(self, xs):
        return [encoder(x) for encoder, x in zip(self.encoders, xs)]

    def reconstruct(self, xs):
        zs = self.encode(xs)
        return [decoder(z) for decoder, z in zip(self.decoders, zs)], zs

    def compute_fusion_weight(self, z1, z2):
        norm1, norm2 = z1.norm(dim=1, keepdim=True), z2.norm(dim=1, keepdim=True)
        denominator = norm1 + norm2 + 1e-8
        return norm1 / denominator, norm2 / denominator

    def fuse_views(self, zs):
        fused = []
        for i in range(self.view):
            for j in range(i + 1, self.view):
                w1, w2 = self.compute_fusion_weight(zs[i], zs[j])
                fused.append(w1 * zs[i] + w2 * zs[j])
        return fused

    def feature_fusion(self, zs, zs_gradient=True):
        inputs = zs if zs_gradient else [z.detach() for z in zs]
        return normalize(self.semantic_fusion(inputs), dim=1)

    def forward(self, xs, zs_gradient=True, return_aux=False, heldout_view=None):
        xrs, zs = self.reconstruct(xs)
        rs = [normalize(self.common_information_module(z), dim=1) for z in zs]
        new_rs = [normalize(self.common_information_module(z), dim=1)
                  for z in self.fuse_views(zs)]
        inputs = zs if zs_gradient else [z.detach() for z in zs]
        raw_H = self.semantic_fusion(inputs)
        H = normalize(raw_H, dim=1)
        output = (xrs, zs, rs, H, new_rs)
        if not return_aux:
            return output  # Original five-value forward interface preserved.
        if not self.training or not torch.is_grad_enabled():
            raise RuntimeError("Auxiliary training losses require train mode and gradients")
        if heldout_view is None:
            heldout_view = int(torch.randint(self.view, ()).item())
        if not 0 <= heldout_view < self.view:
            raise ValueError("Invalid held-out view")
        observed = [v for v in range(self.view) if v != heldout_view]
        # Held-out code is absent from BOTH initialization and inner energy.
        masked_H = self.semantic_fusion(inputs, observed=observed)
        aux = {
            "fit": self.semantic_fusion.fit_loss(raw_H, inputs),
            "cv": self.semantic_fusion.fit_loss(masked_H, inputs, [heldout_view]),
            "heldout_view": heldout_view,
            "pair_weights": self.view_attention(new_rs, H),
        }
        return output + (aux,)
