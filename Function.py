import os

import numpy as np
import timm
import torch
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from torch import optim
from torch import nn
from torch.utils.data import Dataset
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn.functional as F
def getDataLoader(batchsize):
    transform = transforms.Compose( [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5,
    0.5))])

    # 训练集
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchsize, shuffle=True)
    # 测试集
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batchsize, shuffle=False)

    return trainloader, testloader

class CustomCIFAR10Dataset(Dataset):
    def __init__(self, cifar10_dataset, num_classes=10):
        self.cifar10_dataset = cifar10_dataset
        self.num_classes = num_classes

    def __len__(self):
        return len(self.cifar10_dataset)

    def __getitem__(self, idx):
        image, label = self.cifar10_dataset[idx]
        one_hot_label = F.one_hot(torch.tensor(label), num_classes=self.num_classes).float()
        return image, one_hot_label

def one_hotDataLoader(batchsize):
    transform = transforms.Compose( [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5,
    0.5))])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    train_dataset = CustomCIFAR10Dataset(trainset)
    test_dataset = CustomCIFAR10Dataset(testset)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batchsize, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batchsize,shuffle=False)

    return train_loader, test_loader

def train(model, loss, train_dataloader, valid_dataloader, device, batch_size, num_epoch, lr, lr_min, l2_lambda,
          optim='sgd',
          init=True, scheduler_type='Cosine', writer=None,activation=''):
    def init_xavier(m):
        if type(m) == nn.Linear:
            nn.init.xavier_normal_(m.weight)

    if init:
        model.apply(init_xavier)

    model.to(device)

    if optim == 'sgd':
        optimizer = torch.optim.SGD((param for param in model.parameters() if param.requires_grad), lr=lr,
                                    weight_decay=l2_lambda)
    elif optim == 'adam':
        optimizer = torch.optim.Adam((param for param in model.parameters() if param.requires_grad), lr=lr,
                                     weight_decay=l2_lambda)
    elif optim == 'adagrad':
        optimizer = torch.optim.Adagrad((param for param in model.parameters() if param.requires_grad), lr=lr,
                                        weight_decay=l2_lambda)
    elif optim == 'adadelta':
        optimizer = torch.optim.Adadelta((param for param in model.parameters() if param.requires_grad), lr=lr,
                                         weight_decay=l2_lambda)
    elif optim == 'rmsprop':
        optimizer = torch.optim.RMSprop((param for param in model.parameters() if param.requires_grad), lr=lr,
                                        weight_decay=l2_lambda)
    elif optim == 'adamw':
        optimizer = torch.optim.AdamW((param for param in model.parameters() if param.requires_grad), lr=lr,
                                      weight_decay=l2_lambda)
    elif optim == 'nadam':
        optimizer = torch.optim.NAdam((param for param in model.parameters() if param.requires_grad), lr=lr,
                                      weight_decay=l2_lambda)
    elif optim == 'adamax':
        optimizer = torch.optim.Adamax((param for param in model.parameters() if param.requires_grad), lr=lr,
                                       weight_decay=l2_lambda)
    else:
        raise ValueError(f"Error: {optim}")

    if scheduler_type == 'Cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=num_epoch, eta_min=lr_min)

    best_loss = float('inf')  # 初始化为无穷大
    for epoch in range(num_epoch):

        print("\n——————第 {} 轮训练开始——————".format(epoch + 1))
        # epoch=10 320样本 batch=32 10个batch
        # 训练开始
        model.train()
        train_acc = 0
        for batch in tqdm(train_dataloader, desc='训练'):
            imgs, targets = batch
            imgs = imgs.to(device)
            targets = targets.to(device)
            output = model(imgs) # 32,10

            Loss = loss(output, targets)
            # 清掉上一次结果 batch(参数)
            optimizer.zero_grad()
            Loss.backward() # 只计算当前batch
            optimizer.step() # 更新

            _, pred = output.max(1)
            num_correct = (pred == targets).sum().item() # batch  (32,)
            acc = num_correct / (batch_size) # epoch 320/32=10
            # 10 11 12 10次32 x/320
            train_acc += acc

        scheduler.step()
        print("epoch: {}, Loss: {}, Acc: {}".format(epoch + 1, Loss.item(), train_acc / len(train_dataloader)))

        if writer:
            writer.add_scalar('train_loss', Loss.item(), epoch + 1)
            writer.add_scalar('train_accuracy', train_acc / len(train_dataloader), epoch + 1)

        # 测试步骤开始
        model.eval() # 不更新参数
        eval_loss = 0
        eval_acc = 0
        with torch.no_grad():
            for imgs, targets in valid_dataloader:
                imgs = imgs.to(device)
                targets = targets.to(device)
                output = model(imgs)
                Loss = loss(output, targets)
                _, pred = output.max(1)
                num_correct = (pred == targets).sum().item()
                eval_loss += Loss.item()
                acc = num_correct / imgs.shape[0]
                eval_acc += acc

            eval_losses = eval_loss / len(valid_dataloader)
            eval_acc = eval_acc / len(valid_dataloader)

            # 保存最优模型
            if eval_losses < best_loss:
                best_loss = eval_losses
                torch.save(model.state_dict(), f'best_optim_{optim}_activation_{activation}_lr_{lr}_lr_min_{lr_min}_l2_{l2_lambda}.pth')

            print("test Loss: {}".format(eval_losses))
            print("test acc: {}".format(eval_acc))

            if writer:
                writer.add_scalar('test_loss', eval_losses, epoch + 1)
                writer.add_scalar('test_accuracy', eval_acc, epoch + 1)


def SGD(value_and_grad, x, itr, state=None, step_size=0.1, mass=0.9):
    velocity = state if state is not None else np.zeros(len(x))
    val, g = value_and_grad(x)
    velocity = mass * velocity - (1.0 - mass) * g
    x = x + step_size * velocity
    return x, val, g, velocity


def Adam(value_and_grad, x, itr, state=None, step_size=0.001, b1=0.9, b2=0.999, eps=10 ** -8):
    m, v = (np.zeros(len(x)), np.zeros(len(x))) if state is None else state
    val, g = value_and_grad(x)
    m = (1 - b1) * g + b1 * m
    v = (1 - b2) * (g ** 2) + b2 * v
    mhat = m / (1 - b1 ** (itr + 1))
    vhat = v / (1 - b2 ** (itr + 1))
    x = x - (step_size * mhat) / (np.sqrt(vhat) + eps)
    return x, val, g, (m, v)

def train1(model, loss, train_dataloader, valid_dataloader, device, batch_size, num_epoch, lr, lr_min, l2_lambda,
          optim='sgd',
          init=True, scheduler_type='Cosine', writer=None,activation=''):
    def init_xavier(m):
        if type(m) == nn.Linear:
            nn.init.xavier_normal_(m.weight)

    if init:
        model.apply(init_xavier)

    model.to(device)

    if optim == 'sgd':
        optimizer = torch.optim.SGD((param for param in model.parameters() if param.requires_grad), lr=lr,
                                    weight_decay=l2_lambda)
    elif optim == 'adam':
        optimizer = torch.optim.Adam((param for param in model.parameters() if param.requires_grad), lr=lr,
                                     weight_decay=l2_lambda)
    elif optim == 'adagrad':
        optimizer = torch.optim.Adagrad((param for param in model.parameters() if param.requires_grad), lr=lr,
                                        weight_decay=l2_lambda)
    elif optim == 'adadelta':
        optimizer = torch.optim.Adadelta((param for param in model.parameters() if param.requires_grad), lr=lr,
                                         weight_decay=l2_lambda)
    elif optim == 'rmsprop':
        optimizer = torch.optim.RMSprop((param for param in model.parameters() if param.requires_grad), lr=lr,
                                        weight_decay=l2_lambda)
    elif optim == 'adamw':
        optimizer = torch.optim.AdamW((param for param in model.parameters() if param.requires_grad), lr=lr,
                                      weight_decay=l2_lambda)
    elif optim == 'nadam':
        optimizer = torch.optim.NAdam((param for param in model.parameters() if param.requires_grad), lr=lr,
                                      weight_decay=l2_lambda)
    elif optim == 'adamax':
        optimizer = torch.optim.Adamax((param for param in model.parameters() if param.requires_grad), lr=lr,
                                       weight_decay=l2_lambda)
    else:
        raise ValueError(f"Error: {optim}")

    if scheduler_type == 'Cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=num_epoch, eta_min=lr_min)

    best_loss = float('inf')  # 初始化为无穷大
    for epoch in range(num_epoch):

        print("\n——————第 {} 轮训练开始——————".format(epoch + 1))

        # 训练开始
        model.train()
        train_acc = 0
        for batch in tqdm(train_dataloader, desc='训练'):
            imgs, targets = batch
            imgs = imgs.to(device)
            targets = targets.to(device)
            output = model(imgs)

            Loss = loss(output, targets)

            optimizer.zero_grad()
            Loss.backward()
            optimizer.step()

            _, pred = output.max(1)
            num_correct = (pred == targets).sum().item()
            acc = num_correct / (batch_size)
            train_acc += acc

        scheduler.step()
        print("epoch: {}, Loss: {}, Acc: {}".format(epoch + 1, Loss.item(), train_acc / len(train_dataloader)))

        if writer:
            writer.add_scalar('train_loss', Loss.item(), epoch + 1)
            writer.add_scalar('train_accuracy', train_acc / len(train_dataloader), epoch + 1)

        # 测试步骤开始
        model.eval()
        eval_loss = 0
        eval_acc = 0
        with torch.no_grad():
            for imgs, targets in valid_dataloader:
                imgs = imgs.to(device)
                targets = targets.to(device)
                output = model(imgs)
                Loss = loss(output, targets)
                _, pred = output.max(1)
                num_correct = (pred == targets).sum().item()
                eval_loss += Loss.item()
                acc = num_correct / imgs.shape[0]
                eval_acc += acc

            eval_losses = eval_loss / len(valid_dataloader)
            eval_acc = eval_acc / len(valid_dataloader)

            # 保存最优模型
            if eval_losses < best_loss:
                best_loss = eval_losses
                torch.save(model.state_dict(), f'best1.pth')

            print("test Loss: {}".format(eval_losses))
            print("test acc: {}".format(eval_acc))

            if writer:
                writer.add_scalar('test_loss', eval_losses, epoch + 1)
                writer.add_scalar('test_accuracy', eval_acc, epoch + 1)

def train2(model, loss, train_dataloader, valid_dataloader, device, batch_size, num_epoch, lr, lr_min, l2_lambda,
          optim='sgd',
          init=True, scheduler_type='Cosine', writer=None,activation='',criterion=''):
    def init_xavier(m):
        if type(m) == nn.Linear:
            nn.init.xavier_normal_(m.weight)

    if init:
        model.apply(init_xavier)

    model.to(device)

    if optim == 'sgd':
        optimizer = torch.optim.SGD((param for param in model.parameters() if param.requires_grad), lr=lr,
                                    weight_decay=l2_lambda)
    elif optim == 'adam':
        optimizer = torch.optim.Adam((param for param in model.parameters() if param.requires_grad), lr=lr,
                                     weight_decay=l2_lambda)
    elif optim == 'adagrad':
        optimizer = torch.optim.Adagrad((param for param in model.parameters() if param.requires_grad), lr=lr,
                                        weight_decay=l2_lambda)
    elif optim == 'adadelta':
        optimizer = torch.optim.Adadelta((param for param in model.parameters() if param.requires_grad), lr=lr,
                                         weight_decay=l2_lambda)
    elif optim == 'rmsprop':
        optimizer = torch.optim.RMSprop((param for param in model.parameters() if param.requires_grad), lr=lr,
                                        weight_decay=l2_lambda)
    elif optim == 'adamw':
        optimizer = torch.optim.AdamW((param for param in model.parameters() if param.requires_grad), lr=lr,
                                      weight_decay=l2_lambda)
    elif optim == 'nadam':
        optimizer = torch.optim.NAdam((param for param in model.parameters() if param.requires_grad), lr=lr,
                                      weight_decay=l2_lambda)
    elif optim == 'adamax':
        optimizer = torch.optim.Adamax((param for param in model.parameters() if param.requires_grad), lr=lr,
                                       weight_decay=l2_lambda)
    else:
        raise ValueError(f"Error: {optim}")

    if scheduler_type == 'Cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=num_epoch, eta_min=lr_min)

    best_loss = float('inf')  # 初始化为无穷大
    for epoch in range(num_epoch):

        print("\n——————第 {} 轮训练开始——————".format(epoch + 1))

        # 训练开始
        model.train()
        train_acc = 0
        for batch in tqdm(train_dataloader, desc='训练'):
            imgs, targets = batch
            imgs = imgs.to(device)
            targets = targets.to(device)
            output = model(imgs)

            Loss = loss(output, targets)

            optimizer.zero_grad()
            Loss.backward()
            optimizer.step()

            _, pred = output.max(1)
            num_correct = (pred == targets).sum().item()
            acc = num_correct / (batch_size)
            train_acc += acc

        scheduler.step()
        print("epoch: {}, Loss: {}, Acc: {}".format(epoch + 1, Loss.item(), train_acc / len(train_dataloader)))

        if writer:
            writer.add_scalar('train_loss', Loss.item(), epoch + 1)
            writer.add_scalar('train_accuracy', train_acc / len(train_dataloader), epoch + 1)

        # 测试步骤开始
        model.eval()
        eval_loss = 0
        eval_acc = 0
        with torch.no_grad():
            for imgs, targets in valid_dataloader:
                imgs = imgs.to(device)
                targets = targets.to(device)
                output = model(imgs)
                Loss = loss(output, targets)
                _, pred = output.max(1)
                num_correct = (pred == targets).sum().item()
                eval_loss += Loss.item()
                acc = num_correct / imgs.shape[0]
                eval_acc += acc

            eval_losses = eval_loss / len(valid_dataloader)
            eval_acc = eval_acc / len(valid_dataloader)

            # 保存最优模型
            if eval_losses < best_loss:
                best_loss = eval_losses
                torch.save(model.state_dict(), f'best_{criterion}_optim_{optim}_activation_{activation}_lr_{lr}_lr_min_{lr_min}_l2_{l2_lambda}.pth')

            print("test Loss: {}".format(eval_losses))
            print("test acc: {}".format(eval_acc))

            if writer:
                writer.add_scalar('test_loss', eval_losses, epoch + 1)
                writer.add_scalar('test_accuracy', eval_acc, epoch + 1)

def train3(model, loss, train_dataloader, valid_dataloader, device, batch_size, num_epoch, lr, lr_min, l2_lambda,
          optim='sgd',
          init=True, scheduler_type='Cosine', writer=None,activation='',criterion=''):
    def init_xavier(m):
        if type(m) == nn.Linear:
            nn.init.xavier_normal_(m.weight)

    if init:
        model.apply(init_xavier)

    model.to(device)

    if optim == 'sgd':
        optimizer = torch.optim.SGD((param for param in model.parameters() if param.requires_grad), lr=lr,
                                    weight_decay=l2_lambda)
    elif optim == 'adam':
        optimizer = torch.optim.Adam((param for param in model.parameters() if param.requires_grad), lr=lr,
                                     weight_decay=l2_lambda)
    elif optim == 'adagrad':
        optimizer = torch.optim.Adagrad((param for param in model.parameters() if param.requires_grad), lr=lr,
                                        weight_decay=l2_lambda)
    elif optim == 'adadelta':
        optimizer = torch.optim.Adadelta((param for param in model.parameters() if param.requires_grad), lr=lr,
                                         weight_decay=l2_lambda)
    elif optim == 'rmsprop':
        optimizer = torch.optim.RMSprop((param for param in model.parameters() if param.requires_grad), lr=lr,
                                        weight_decay=l2_lambda)
    elif optim == 'adamw':
        optimizer = torch.optim.AdamW((param for param in model.parameters() if param.requires_grad), lr=lr,
                                      weight_decay=l2_lambda)
    elif optim == 'nadam':
        optimizer = torch.optim.NAdam((param for param in model.parameters() if param.requires_grad), lr=lr,
                                      weight_decay=l2_lambda)
    elif optim == 'adamax':
        optimizer = torch.optim.Adamax((param for param in model.parameters() if param.requires_grad), lr=lr,
                                       weight_decay=l2_lambda)
    else:
        raise ValueError(f"Error: {optim}")

    if scheduler_type == 'Cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=num_epoch, eta_min=lr_min)

    best_loss = float('inf')  # 初始化为无穷大
    for epoch in range(num_epoch):

        print("\n——————第 {} 轮训练开始——————".format(epoch + 1))

        # 训练开始
        model.train()
        train_acc = 0
        for batch in tqdm(train_dataloader, desc='训练'):
            imgs, targets = batch
            imgs = imgs.to(device)
            targets = targets.to(device)
            targets_one_hot = F.one_hot(targets, num_classes=10).float()
            output = model(imgs)

            Loss = loss(output, targets_one_hot)

            optimizer.zero_grad()
            Loss.backward()
            optimizer.step()

            _, pred = output.max(1)
            num_correct = (pred == targets).sum().item()
            acc = num_correct / (batch_size)
            train_acc += acc

        scheduler.step()
        print("epoch: {}, Loss: {}, Acc: {}".format(epoch + 1, Loss.item(), train_acc / len(train_dataloader)))

        if writer:
            writer.add_scalar('train_loss', Loss.item(), epoch + 1)
            writer.add_scalar('train_accuracy', train_acc / len(train_dataloader), epoch + 1)

        # 测试步骤开始
        model.eval()
        eval_loss = 0
        eval_acc = 0
        with torch.no_grad():
            for imgs, targets in valid_dataloader:
                imgs = imgs.to(device)
                targets = targets.to(device)
                targets_one_hot = F.one_hot(targets, num_classes=10).float()
                output = model(imgs)
                Loss = loss(output, targets_one_hot)
                _, pred = output.max(1)
                num_correct = (pred == targets).sum().item()
                eval_loss += Loss.item()
                acc = num_correct / imgs.shape[0]
                eval_acc += acc

            eval_losses = eval_loss / len(valid_dataloader)
            eval_acc = eval_acc / len(valid_dataloader)

            # 保存最优模型
            if eval_losses < best_loss:
                best_loss = eval_losses
                torch.save(model.state_dict(), f'best_{criterion}_optim_{optim}_activation_{activation}_lr_{lr}_lr_min_{lr_min}_l2_{l2_lambda}.pth')

            print("test Loss: {}".format(eval_losses))
            print("test acc: {}".format(eval_acc))

            if writer:
                writer.add_scalar('test_loss', eval_losses, epoch + 1)
                writer.add_scalar('test_accuracy', eval_acc, epoch + 1)