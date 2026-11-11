import argparse
import torch
from network import Network
from metric import valid
from dataloader import load_data


def main():
    parser = argparse.ArgumentParser(description='Evaluate saved VEISI checkpoint')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--dataset', default=None)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--kmeans_n_init', type=int, default=100)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--eval_views', action='store_true')
    args = parser.parse_args()
    device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu')
                          if args.device == 'auto' else args.device)
    # Saved payload contains only tensors and primitive containers.
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    if 'model_config' not in checkpoint:
        raise ValueError('Original VEISI checkpoints are incompatible; retrain the new model')
    name = args.dataset or checkpoint['args']['dataset']
    dataset, dims, view, size, classes = load_data(name)
    config = checkpoint['model_config']
    if config['input_size'] != dims or config['view'] != view:
        raise ValueError('Dataset dimensions differ from checkpoint')
    model = Network(device=device, **config)
    model.load_state_dict(checkpoint['model_state_dict'])
    valid(model, device, dataset, view, size, classes, eval_h=args.eval_views,
          batch_size=args.batch_size, n_init=args.kmeans_n_init,
          random_state=checkpoint['args']['seed'])


if __name__ == '__main__':
    main()
