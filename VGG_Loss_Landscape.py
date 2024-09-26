import matplotlib as mpl

mpl.use('Agg')  # 使用非GUI后端
import matplotlib.pyplot as plt
from torch import nn
import numpy as np
import torch
import os
import random
from tqdm import tqdm as tqdm
from IPython import display
from VGG_BatchNorm.models.vgg import VGG_A
from VGG_BatchNorm.models.vgg import VGG_BatchNorm
# from src.models.vgg import VGG_A_BatchNorm
from VGG_BatchNorm.data.loaders import get_cifar_loader


# 计算模型分类准确度的函数
def get_accuracy(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():  # 禁用梯度计算，加速评估过程
        for data in dataloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)  # 将数据传输到设备
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)  # 预测结果
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total


# 设置随机种子以确保结果可重复
def set_random_seeds(seed_value=0, device='cpu'):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    random.seed(seed_value)
    if device != 'cpu':
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# 完整的训练过程函数，记录每一步的损失值
def train(model, optimizer, criterion, train_loader, val_loader, scheduler=None, epochs_n=100, best_model_path=None):
    model.to(device)  # 将模型传输到设备
    learning_curve = [np.nan] * epochs_n  # 学习曲线
    train_accuracy_curve = [np.nan] * epochs_n  # 训练准确度曲线
    val_accuracy_curve = [np.nan] * epochs_n  # 验证准确度曲线
    max_val_accuracy = 0
    max_val_accuracy_epoch = 0

    batches_n = len(train_loader)  # 批次数量
    losses_list = []
    grads = []
    for epoch in tqdm(range(epochs_n), unit='epoch'):  # 迭代每个epoch
        if scheduler is not None:
            scheduler.step()  # 更新学习率
        model.train()  # 设定模型为训练模式

        loss_list = []  # 记录每一步的损失值
        grad_list = []  # 记录每一步的梯度
        learning_curve[epoch] = 0  # 初始化当前epoch的损失

        for data in train_loader:
            x, y = data
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()  # 清除上一步的梯度
            prediction = model(x)
            loss = criterion(prediction, y)
            # 记录当前步骤的损失值
            loss_list.append(loss.item())
            # 计算梯度
            loss.backward()
            # 记录梯度（例如，对于某一层的权重）
            grad_list = model.classifier[4].weight.grad.clone().cpu().numpy()
            optimizer.step()  # 更新模型参数
            # 更新学习曲线
            learning_curve[epoch] += loss.item()

        # 存储每个epoch的损失和梯度
        losses_list.append(loss_list)
        grads.append(grad_list)

        # 绘制当前学习曲线
        display.clear_output(wait=True)
        f, axes = plt.subplots(1, 2, figsize=(15, 3))
        learning_curve[epoch] /= batches_n
        axes[0].plot(learning_curve, label='Learning Curve')
        axes[0].set_title('Learning Curve')
        axes[0].legend()

    return losses_list, grads


if __name__ == '__main__':
    # ## 常量（参数）初始化
    device_id = [0, 1, 2, 3]
    num_workers = 4
    batch_size = 128

    # 添加包目录到路径
    module_path = os.path.dirname(os.getcwd())
    home_path = module_path
    figures_path = os.path.join(home_path, 'reports', 'figures')
    models_path = os.path.join(home_path, 'reports', 'models')

    # 确保使用正确的设备
    device_id = device_id
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    device = torch.device("cuda:{}".format(0) if torch.cuda.is_available() else "cpu")
    print(device)
    print(torch.cuda.get_device_name(0))

    # Data Loader
    train_loader = get_cifar_loader(train=True)
    val_loader = get_cifar_loader(train=False)
    for X, y in train_loader:
        # 打印输入和目标的形状
        print("Input shape:", X.shape)
        print("Target shape:", y.shape)

        # 可视化样本图像
        img = X[0].numpy().transpose(1, 2, 0)
        img = (img - img.min()) / (img.max() - img.min())  # 归一化到[0, 1]
        plt.imshow(img)
        plt.title("Label: {y[0]}")
        plt.savefig('sample_image.png')  # 将样本图像保存到文件
        plt.close()  # 关闭图像以避免显示问题
        break

    # 训练您的模型
    epo = 20
    loss_save_path = ''
    grad_save_path = ''

    set_random_seeds(seed_value=2020, device=device)
    # 训练 VGG_A 模型
    model_a1 = VGG_A()
    model_a2 = VGG_A()
    model_a3 = VGG_A()
    model_a4 = VGG_A()
    model_bn1 = VGG_BatchNorm()
    model_bn2 = VGG_BatchNorm()
    model_bn3 = VGG_BatchNorm()
    model_bn4 = VGG_BatchNorm()
    optimizer_a1 = torch.optim.Adam(model_a1.parameters(), lr=1e-3)
    optimizer_a2 = torch.optim.Adam(model_a2.parameters(), lr=2e-3)
    optimizer_a3 = torch.optim.Adam(model_a3.parameters(), lr=1e-4)
    optimizer_a4 = torch.optim.Adam(model_a4.parameters(), lr=5e-4)
    optimizer_bn1 = torch.optim.Adam(model_bn1.parameters(), lr=1e-3)
    optimizer_bn2 = torch.optim.Adam(model_bn2.parameters(), lr=1e-3)
    optimizer_bn3 = torch.optim.Adam(model_bn3.parameters(), lr=1e-3)
    optimizer_bn4 = torch.optim.Adam(model_bn4.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    losses_a1, grads_a1 = train(model_a1, optimizer_a1, criterion, train_loader, val_loader, epochs_n=epo)
    losses_a2, grads_a2 = train(model_a2, optimizer_a2, criterion, train_loader, val_loader, epochs_n=epo)
    losses_a3, grads_a3 = train(model_a3, optimizer_a3, criterion, train_loader, val_loader, epochs_n=epo)
    losses_a4, grads_a4 = train(model_a4, optimizer_a4, criterion, train_loader, val_loader, epochs_n=epo)
    losses_bn1, grads_bn1 = train(model_bn1, optimizer_bn1, criterion, train_loader, val_loader, epochs_n=epo)
    losses_bn2, grads_bn2 = train(model_bn2, optimizer_bn2, criterion, train_loader, val_loader, epochs_n=epo)
    losses_bn3, grads_bn3 = train(model_bn3, optimizer_bn3, criterion, train_loader, val_loader, epochs_n=epo)
    losses_bn4, grads_bn4 = train(model_bn4, optimizer_bn4, criterion, train_loader, val_loader, epochs_n=epo)

    np.savetxt(os.path.join(loss_save_path, 'loss_a1.txt'), losses_a1, fmt='%s', delimiter=' ')
    np.savetxt(os.path.join(loss_save_path, 'loss_a2.txt'), losses_a2, fmt='%s', delimiter=' ')
    np.savetxt(os.path.join(loss_save_path, 'loss_a3.txt'), losses_a3, fmt='%s', delimiter=' ')
    np.savetxt(os.path.join(loss_save_path, 'loss_a4.txt'), losses_a4, fmt='%s', delimiter=' ')
    np.savetxt(os.path.join(loss_save_path, 'loss_bn1.txt'), losses_bn1, fmt='%s', delimiter=' ')
    np.savetxt(os.path.join(loss_save_path, 'loss_bn2.txt'), losses_bn2, fmt='%s', delimiter=' ')
    np.savetxt(os.path.join(loss_save_path, 'loss_bn3.txt'), losses_bn3, fmt='%s', delimiter=' ')
    np.savetxt(os.path.join(loss_save_path, 'loss_bn4.txt'), losses_bn4, fmt='%s', delimiter=' ')

    # 将所有损失值展平
    flat_losses_a1 = [loss for epoch_losses in losses_a1 for loss in epoch_losses]
    flat_losses_a2 = [loss for epoch_losses in losses_a2 for loss in epoch_losses]
    flat_losses_a3 = [loss for epoch_losses in losses_a3 for loss in epoch_losses]
    flat_losses_a4 = [loss for epoch_losses in losses_a4 for loss in epoch_losses]
    flat_losses_bn1 = [loss for epoch_losses in losses_bn1 for loss in epoch_losses]
    flat_losses_bn2 = [loss for epoch_losses in losses_bn2 for loss in epoch_losses]
    flat_losses_bn3 = [loss for epoch_losses in losses_bn3 for loss in epoch_losses]
    flat_losses_bn4 = [loss for epoch_losses in losses_bn4 for loss in epoch_losses]

    # 计算每一步的最小值和最大值
    min_curve_a = []
    max_curve_a = []
    for step_losses in zip(flat_losses_a1, flat_losses_a2, flat_losses_a3, flat_losses_a4):
        min_curve_a.append(min(step_losses))
        max_curve_a.append(max(step_losses))

    min_curve_bn = []
    max_curve_bn = []
    for step_losses in zip(flat_losses_bn1, flat_losses_bn2, flat_losses_bn3, flat_losses_bn4):
        min_curve_bn.append(min(step_losses))
        max_curve_bn.append(max(step_losses))


    # 绘制损失曲线
    def plot_loss_curve(losses, label, color):
        plt.plot(range(len(losses)), losses, label=label, color=color, linewidth=0.2)


    def plot_min_max_curve():
        plt.figure(figsize=(10, 5))  # 增加图形宽度
        plt.plot(range(len(min_curve_a)), min_curve_a, color='green', linewidth=0.1)
        plt.plot(range(len(max_curve_a)), max_curve_a, color='green', linewidth=0.1)
        plt.fill_between(range(len(min_curve_a)), min_curve_a, max_curve_a, label='VGG_A', alpha=0.4, color='green')
        plt.plot(range(len(min_curve_bn)), min_curve_bn, color='red', linewidth=0.1)
        plt.plot(range(len(max_curve_bn)), max_curve_bn, color='red', linewidth=0.1)
        plt.fill_between(range(len(min_curve_bn)), min_curve_bn, max_curve_bn, label='VGG_A with BN', alpha=0.4,
                         color='red')
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        plt.title('Loss landscape')
        plt.legend()
        plt.savefig('Loss landscape.png')  # 保存图像到文件


    # 绘制各个优化器的损失曲线
    plt.figure(figsize=(10, 5))
    plot_loss_curve(flat_losses_a1, 'VGG_A Loss (lr=1e-3)', 'blue')
    plot_loss_curve(flat_losses_a2, 'VGG_A Loss (lr=2e-3)', 'green')
    plot_loss_curve(flat_losses_a3, 'VGG_A Loss (lr=1e-4)', 'red')
    plot_loss_curve(flat_losses_a4, 'VGG_A Loss (lr=5e-4)', 'purple')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.title('Loss')
    plt.legend()
    plt.savefig('Loss without BN.png')  # 保存图像到文件

    # 绘制各个优化器的损失曲线
    plt.figure(figsize=(10, 5))
    plot_loss_curve(flat_losses_bn1, 'VGG_A with BN Loss (lr=1e-3)', 'blue')
    plot_loss_curve(flat_losses_bn2, 'VGG_A with BN Loss (lr=2e-3)', 'green')
    plot_loss_curve(flat_losses_bn3, 'VGG_A with BN Loss (lr=1e-4)', 'red')
    plot_loss_curve(flat_losses_bn4, 'VGG_A with BN Loss (lr=5e-4)', 'purple')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.title('Loss')
    plt.legend()
    plt.savefig('Loss with BN.png')  # 保存图像到文件

    # 绘制最小值和最大值曲线
    plot_min_max_curve()
