import argparse
import copy
import os
import torch
from copy import deepcopy
from torch import nn
from torchvision import datasets, transforms
import numpy as np
from models.LeNet5 import LeNet5
from eval.metrics import get_accuracy
from bitarray import bitarray
from models.YourNet import TeacherNet, YourNet
from matplotlib import pyplot as plt
import torch.nn.functional as F

parser = argparse.ArgumentParser()

parser.add_argument('--checkpoint-dir', type=str, required=True)
parser.add_argument('--last-checkpoint', type=str, default=None)

parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], default='cpu')
parser.add_argument('--batch-size', type=int, default=64)
parser.add_argument('--epoch-start', type=int, default=0)
parser.add_argument('--epoch-end', type=int, required=True)

args = parser.parse_args()


def prune(model: nn.Module, perc):
    masks = []
    for parameter in model.parameters():
        if len(parameter.data.size()) != 1:
            weight = parameter.data.abs().numpy().flatten()
            pruned = parameter.data.abs() > np.percentile(weight, perc)
            masks.append(pruned.float())
    return masks


def distillation(y, labels, teacher_scores, temp, alpha):
    return nn.KLDivLoss()(F.log_softmax(y / temp, dim=1), F.softmax(teacher_scores / temp, dim=1)) * (
        alpha) + F.cross_entropy(y, labels) * (1. - alpha)


if __name__ == '__main__':
    ###################### Begin #########################
    # You can run your train() here
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(root='./data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(root='./data', train=False,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True)

    teacher_model = TeacherNet().to(device=args.device)
    if args.last_checkpoint is not None:
        teacher_model.load_state_dict(torch.load(args.last_checkpoint, map_location=args.device))

    # optimizer = torch.optim.SGD(model.parameters(), lr=0.008)
    optimizer_teacher = torch.optim.Adadelta(teacher_model.parameters(), lr=0.02)
    loss_fn = nn.CrossEntropyLoss()
    acc_teacher = 0
    best_teacher = teacher_model
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
    if args.last_checkpoint is None:
        for epoch in range(args.epoch_start, args.epoch_end):
            print(f"Epoch {epoch}\n-------------------------------")
            size = len(train_loader.dataset)
            teacher_model.train()
            for batch_idx, (X, y) in enumerate(train_loader):

                X, y = X.to(args.device), y.to(args.device)

                # Compute prediction error
                pred_y = teacher_model(X)
                loss = loss_fn(pred_y, y)

                # Backpropagation
                optimizer_teacher.zero_grad()
                loss.backward()
                optimizer_teacher.step()

                if batch_idx % 100 == 0:
                    loss, current = loss.item(), batch_idx * len(X)
                    print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

            accuracy = get_accuracy(teacher_model, test_loader, args.device)
            print("Accuracy: %.3f}" % accuracy)
            if accuracy > acc_teacher:
                acc_teacher = accuracy
                best_teacher = copy.deepcopy(teacher_model)

            torch.save(teacher_model.state_dict(), args.checkpoint_dir + f'teacher-epoch-{epoch}.pth')
    torch.save(best_teacher.state_dict(), args.checkpoint_dir + f'best-teacher.pth')

    model = YourNet().to(device=args.device)
    optimizer = torch.optim.Adadelta(model.parameters(), lr=0.05)
    best_acc = 0
    best_model = model
    for epoch in range(args.epoch_start, args.epoch_end):
        print(f"Epoch {epoch}\n-------------------------------")
        size = len(train_loader.dataset)
        model.train()
        for batch_idx, (X, y) in enumerate(train_loader):
            X, y = X.to(args.device), y.to(args.device)
            # Compute prediction error
            pred_y = model(X)
            pred_teacher = teacher_model(X).detach()
            loss = distillation(pred_y, y, pred_teacher, temp=10.0, alpha=0.65)
            # loss = loss_fn(pred_y, y)
            optimizer.zero_grad()

            # Backpropagation
            loss.backward()
            optimizer.step()
            if batch_idx % 100 == 0:
                loss, current = loss.item(), batch_idx * len(X)
                print(f"student loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

        accuracy = get_accuracy(model, test_loader, args.device)
        if accuracy > best_acc:
            best_acc = accuracy
            best_model = copy.deepcopy(model)

        print("Accuracy: %.3f}" % accuracy)

        torch.save(model.state_dict(), args.checkpoint_dir + f'epoch-{epoch}.pth')
    torch.save(best_model.state_dict(), args.checkpoint_dir + f'best.pth')
    # after_prune = deepcopy(model)
    # masks = prune(model, 60)
    # masks = prune(model,60)
    # after_prune.set_masks(masks)
    # print("-----------------------------------------------------------------------------------")
    # torch.save(model.state_dict(), args.checkpoint_dir + 'epoch-quant-before.pth')
    #
    # quant_model = deepcopy(model)
    # plot_weights(model)
    # masks = prune(quant_model, 30)
    # quant_model.set_masks(masks)
    # quant_model.kmeans_quant(bits=4)
    # plot_weights(quant_model)
    #
    # torch.save(quant_model.state_dict(), args.checkpoint_dir + 'epoch-quant.pth')
    print("-----------------------------------------------------------------------------------")

    ######################  End  #########################
