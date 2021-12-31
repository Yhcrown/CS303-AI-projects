"""
Test LeNet5

Example:
python test_lenet5.py \
  --best-checkpoint ./checkpoints/LeNet5/epoch-6.pth \
  --device cpu
"""
import argparse
import torch

from torchvision import datasets, transforms
from models.LeNet5 import LeNet5
from eval.metrics import get_accuracy, get_infer_time, get_macs_and_params

# --------------- Arguments ---------------

parser = argparse.ArgumentParser()

parser.add_argument('--best-checkpoint', type=str, required=True)

parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], default='cpu')
parser.add_argument('--batch-size', type=int, default=64)

args = parser.parse_args()


# --------------- Loading ---------------


if __name__ == '__main__':

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(root='./data', train=False,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True)

    model = LeNet5().to(device=args.device)
    model.load_state_dict(torch.load(args.best_checkpoint, map_location=args.device))

    accuracy = get_accuracy(model, test_loader, args.device)
    infer_time = get_infer_time(model, test_loader, args.device)
    MACs, params = get_macs_and_params(model, args.device)

    print("----------------------------------------------------------------")
    print("| %10s | %8s | %14s | %9s | %7s |" % ("Model Name","Accuracy", "Infer Time(ms)", "Params(M)", "MACs(M)"))
    print("----------------------------------------------------------------")
    print("| %10s | %8.3f | %14.3f | %9.3f | %7.3f |" % ("LeNet-5", accuracy, infer_time * 1000, MACs / (1000 ** 2), params / (1000 ** 2)))
    print("----------------------------------------------------------------")