import torch
import os
from tensorboardX import SummaryWriter
from torch import optim, nn
from sklearn.metrics import classification_report, confusion_matrix
import torchvision.models as models
from Function import train, getDataLoader
from Model import Model
from torchsummary import summary
import matplotlib.pyplot as plt
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def draw_report(classification_report_text):
    # 创建一个新的图像
    plt.figure(figsize=(8, 6))

    # 使用text函数将分类报告文本绘制到图像上
    plt.text(0.05, 0.5, classification_report_text, fontsize=10)

    # 隐藏坐标轴
    plt.axis('off')
    # 保存图像为图片文件
    # plt.savefig('classification_report.png')
    plt.show()


def draw_confusion_matrix(conf_matrix,filename):
    # 创建一个新的图像
    plt.figure(figsize=(8, 6))

    # 使用imshow函数绘制混淆矩阵
    plt.imshow(conf_matrix, cmap=plt.cm.Blues)

    # 添加标题和标签
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

    # 显示颜色条
    plt.colorbar()

    # 添加文本标签
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            plt.text(j, i, str(conf_matrix[i, j]), ha='center', va='center')


    # 保存图像为图片文件
    plt.savefig(f'{filename}.png')
    # plt.show()

def test(model, valid_dataloader, loss):
    eval_loss = 0
    eval_acc = 0
    predictions = []
    targets_list = []

    with torch.no_grad():
        total_samples = 0
        correct_predictions = 0
        for imgs, targets in valid_dataloader:
            imgs = imgs.to(device)
            targets = targets.to(device)
            output = model(imgs)
            Loss = loss(output, targets)
            eval_loss += Loss.item()

            _, pred = output.max(1)
            predictions.extend(pred.cpu().numpy())
            targets_list.extend(targets.cpu().numpy())

            total_samples += targets.size(0)
            correct_predictions += (pred == targets).sum().item()

        eval_losses = eval_loss / len(valid_dataloader)
        eval_acc = correct_predictions / total_samples
        print(f"test Loss: {eval_losses}")
        print(f"test acc: {round(eval_acc, 4)}")

    return predictions, targets_list

batch_size = 32
trainloader, testloader = getDataLoader(batchsize=batch_size)
model = Model(nn.ReLU).to(device)
model_path = 'best_{\'NLLLoss\'}_optim_adam_activation_ReLU_lr_0.01_lr_min_1e-05_l2_0.0001.pth'

state_dict = torch.load(model_path)
model.load_state_dict(state_dict)
model.eval()
criterion = nn.CrossEntropyLoss()

# 获取预测结果和真实标签
predictions, targets = test(model, testloader, criterion)

# 生成分类报告
classificationReport = classification_report(targets, predictions,digits=4)

with open('classification_reports.txt', 'a') as file:
    file.write(f"NLLLoss Classification Report:\n")
    file.write(classification_report(targets, predictions, digits=4))
    file.write("\n")

conf_matrix = confusion_matrix(targets, predictions)
draw_confusion_matrix(conf_matrix,"NLLLoss")