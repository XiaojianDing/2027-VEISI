"""CPU integration checks for the VEISI implementation."""
import copy
import io
import torch
from network import Network, IterativeSemanticFusion
from loss import ContrastiveLoss, training_objective


def main():
    torch.set_num_threads(1)
    torch.manual_seed(10)
    config = dict(view=3, input_size=[7, 9, 11], feature_dim=8,
                  high_feature_dim=4, inference_steps=3, predictor_hidden=12, num_heads=2)
    model = Network(device='cpu', **config)
    xs = [torch.randn(6, d) for d in config['input_size']]
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)
    total, terms = training_objective(model, xs, ContrastiveLoss(), heldout_view=1)
    assert set(terms) == {'rec', 'fcl', 'cv', 'fit'}
    assert all(torch.isfinite(t) for t in terms.values())
    total.backward()
    for name, group in [('encoder', model.encoders), ('initializer', model.semantic_fusion.initializer),
                        ('predictor', model.semantic_fusion.predictors), ('attention', model.view_attention)]:
        grads = [p.grad for p in group.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads), name
        assert sum(g.abs().sum().item() for g in grads) > 0, name
    optimizer.step()
    print('PASS: four-loss backward and finite nonzero gradients in all module groups')
    fusion = model.semantic_fusion.eval()
    zs = [torch.randn(6, 8) for _ in range(3)]
    with torch.no_grad():
        h = fusion(zs, observed=[0, 2])
        assert torch.allclose(h, fusion([zs[0], zs[1]*100+50, zs[2]], [0, 2]), atol=1e-6)
        full = fusion(zs)
        split = torch.cat([fusion([z[:2] for z in zs]), fusion([z[2:] for z in zs])])
        assert torch.allclose(full, split, atol=1e-6)
        assert not full.requires_grad
        assert fusion([z[:1] for z in zs]).shape == (1, 4)
        zero = copy.deepcopy(fusion)
        zero.steps = 0
        assert (full - zero(zs)).abs().max() > 1e-6
    print('PASS: held-out exclusion, no_grad inference, singleton, batch invariance, active iteration')
    fusion.train()
    inputs = [torch.randn(6, 8, requires_grad=True) for _ in range(3)]
    cv = fusion.fit_loss(fusion(inputs, [0, 2]), inputs, [1])
    grads = torch.autograd.grad(cv, inputs, allow_unused=True)
    assert grads[1] is None or grads[1].abs().sum() == 0
    assert grads[0] is not None and grads[0].abs().sum() > 0
    print('PASS: held-out target has no cv encoder gradient')
    tiny = IterativeSemanticFusion(2, 3, 2, steps=2, hidden_dim=4).double().train()
    tz = [torch.randn(2, 3, dtype=torch.double) for _ in range(2)]
    parameter = tiny.predictors[0][0].weight
    analytic, = torch.autograd.grad(tiny(tz).square().sum(), parameter)
    eps = 1e-5
    with torch.no_grad():
        saved = parameter[0, 0].item()
        parameter[0, 0] = saved + eps
    plus = tiny(tz).square().sum().item()
    with torch.no_grad():
        parameter[0, 0] = saved - eps
    minus = tiny(tz).square().sum().item()
    with torch.no_grad():
        parameter[0, 0] = saved
    assert abs(analytic[0, 0].item() - (plus-minus)/(2*eps)) < 1e-5
    print('PASS: unrolled second-order gradient agrees with finite differences')
    model.eval()
    with torch.no_grad():
        expected = model(xs)[3]
    buffer = io.BytesIO()
    torch.save({'model_state_dict': model.state_dict(), 'model_config': config}, buffer)
    buffer.seek(0)
    payload = torch.load(buffer, weights_only=True)
    restored = Network(device='cpu', **payload['model_config']).eval()
    restored.load_state_dict(payload['model_state_dict'])
    with torch.no_grad():
        assert torch.allclose(expected, restored(xs)[3])
    two = Network(2, [5, 7], 8, 4, 'cpu', num_heads=2)
    assert len(two([torch.randn(3, 5), torch.randn(3, 7)])) == 5
    print('PASS: checkpoint roundtrip, two views and original forward interface')


if __name__ == '__main__':
    main()
