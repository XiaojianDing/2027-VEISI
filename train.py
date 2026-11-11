import argparse
import json
from pathlib import Path
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from network import Network
from loss import ContrastiveLoss, training_objective
from dataloader import load_data
from metric import valid


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_parser():
    p = argparse.ArgumentParser(description='VEISI with iterative semantic inference')
    p.add_argument('--dataset', default='100leaves')
    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--learning_rate', type=float, default=0.0003)
    p.add_argument('--weight_decay', type=float, default=0.0)
    p.add_argument('--pre_epochs', type=int, default=2)
    p.add_argument('--con_epochs', type=int, default=None)
    p.add_argument('--feature_dim', type=int, default=64)
    p.add_argument('--high_feature_dim', type=int, default=20)
    p.add_argument('--temperature', type=float, default=1.0)
    p.add_argument('--alpha', type=float, default=1.0)
    p.add_argument('--beta', type=float, default=1.0)
    p.add_argument('--gamma', type=float, default=1.0)
    p.add_argument('--inference_steps', type=int, default=3)
    p.add_argument('--inference_lr', type=float, default=0.5)
    p.add_argument('--proximal_lambda', type=float, default=0.1)
    p.add_argument('--robust_c', type=float, default=1.0)
    p.add_argument('--predictor_hidden', type=int, default=128)
    p.add_argument('--num_heads', type=int, default=4)
    p.add_argument('--contrastive_mode', choices=['legacy', 'infonce'], default='legacy')
    p.add_argument('--seed', type=int, default=None)
    p.add_argument('--device', default='auto')
    p.add_argument('--eval_every', type=int, default=1, help='0 disables label-based evaluation')
    p.add_argument('--eval_batch_size', type=int, default=512)
    p.add_argument('--kmeans_n_init', type=int, default=100)
    p.add_argument('--grad_clip', type=float, default=5.0)
    p.add_argument('--output_dir', default='models')
    p.add_argument('--max_batches', type=int, default=0, help='Debug only; 0 uses all batches')
    return p


def main():
    args = build_parser().parse_args()
    if args.batch_size < 2 or args.pre_epochs < 0 or args.eval_every < 0 or args.max_batches < 0:
        raise ValueError('Invalid batch size, epoch count, evaluation interval, or batch limit')
    if min(args.alpha, args.beta, args.gamma) < 0:
        raise ValueError('Loss coefficients must be nonnegative')
    defaults = {'MNIST-USPS': 50, '100leaves': 100, 'BDGP': 8, 'Fashion': 50,
                'Caltech-5V': 100, 'Cifar10': 10, 'Synthetic3d': 100}
    if args.con_epochs is None:
        args.con_epochs = defaults.get(args.dataset, 50)
    if args.con_epochs < 0:
        raise ValueError('con_epochs must be nonnegative')
    if args.seed is None:
        args.seed = {'Caltech-5V': 1000000, 'Synthetic3d': 100}.get(args.dataset, 10)
    setup_seed(args.seed)
    device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu')
                          if args.device == 'auto' else args.device)
    dataset, dims, view, data_size, class_num = load_data(args.dataset)
    if len(dataset) < 2:
        raise ValueError('Need at least two samples')
    # Do not discard whole small datasets; skip singleton remainder for contrast.
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
    config = dict(view=view, input_size=dims, feature_dim=args.feature_dim,
                  high_feature_dim=args.high_feature_dim, inference_steps=args.inference_steps,
                  inference_lr=args.inference_lr, proximal_lambda=args.proximal_lambda,
                  robust_c=args.robust_c, predictor_hidden=args.predictor_hidden,
                  num_heads=args.num_heads)
    model = Network(device=device, **config)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate,
                                 weight_decay=args.weight_decay)
    contrastive = ContrastiveLoss(temperature=args.temperature, mode=args.contrastive_mode)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    history, best_acc = [], -1.0

    def save(name, epoch):
        torch.save({'model_state_dict': model.state_dict(), 'model_config': config,
                    'args': vars(args), 'epoch': epoch, 'history': history,
                    'optimizer_state_dict': optimizer.state_dict()}, out / name)

    for epoch in range(1, args.pre_epochs + args.con_epochs + 1):
        model.train()
        pretraining = epoch <= args.pre_epochs
        totals = {'total': 0.0, 'rec': 0.0, 'fcl': 0.0, 'cv': 0.0, 'fit': 0.0}
        count = 0
        for xs, _, _ in loader:
            xs = [x.to(device) for x in xs]
            if not pretraining and xs[0].shape[0] < 2:
                continue
            optimizer.zero_grad(set_to_none=True)
            if pretraining:
                xrs, _ = model.reconstruct(xs)  # No unnecessary unrolled inference.
                total = torch.stack([torch.nn.functional.mse_loss(a, b)
                                     for a, b in zip(xrs, xs)]).mean()
                terms = {'rec': total}
            else:
                total, terms = training_objective(model, xs, contrastive,
                                                  args.alpha, args.beta, args.gamma)
            if not torch.isfinite(total):
                raise FloatingPointError('Nonfinite loss; reduce inference_lr or learning_rate')
            total.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip,
                                               error_if_nonfinite=True)
            optimizer.step()
            totals['total'] += total.item()
            for name, value in terms.items():
                totals[name] += value.item()
            count += 1
            if args.max_batches and count >= args.max_batches:
                break
        if not count:
            raise RuntimeError('No training batches')
        row = {'epoch': epoch, 'phase': 'pretrain' if pretraining else 'joint',
               **{name: value / count for name, value in totals.items()}}
        print(json.dumps(row), flush=True)
        if not pretraining and args.eval_every and (
                (epoch - args.pre_epochs) % args.eval_every == 0 or
                epoch == args.pre_epochs + args.con_epochs):
            acc, nmi, pur = valid(model, device, dataset, view, data_size, class_num,
                                  epoch=epoch, batch_size=args.eval_batch_size,
                                  n_init=args.kmeans_n_init, random_state=args.seed)
            row.update(acc=acc, nmi=nmi, pur=pur)
            if acc > best_acc:
                best_acc = acc
                history.append(row)
                save(f'{args.dataset}_best_acc.pth', epoch)
                history.pop()
        history.append(row)
        save(f'{args.dataset}_last.pth', epoch)
        (out / f'{args.dataset}_history.json').write_text(
            json.dumps(history, indent=2), encoding='utf-8')
    print('Saved final checkpoint; best_acc checkpoints use ground-truth evaluation labels.')


if __name__ == '__main__':
    main()
