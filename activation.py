import matplotlib.pyplot as plt
import numpy as np
from typing import List  # 导入List类型
import csv

# 读取CSV文件并将第三列数据转换为浮点数
def read_csv_column(filename: str, column_index: int) -> List[float]:
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # 跳过表头行，如果有的话
        column_data = [float(row[column_index]) for row in reader]
    return column_data

activations=["ReLU", "ELU", "PReLU", "SELU","LeakyReLU"]
optims =['sgd', 'adam', 'adagrad', 'adamax', 'rmsprop', 'adamw','nadam','adadelta']

for activation in activations:
    # 读取训练损失和测试损失数据
    d1 = read_csv_column(f'csv/optim_{optims[0]}_activation{activation}_lr_0.01_lr_min1e-05_l2_0.0001.csv', 3)
    d2 = read_csv_column(f'csv/optim_{optims[1]}_activation{activation}_lr_0.01_lr_min1e-05_l2_0.0001.csv',3)
    d3 = read_csv_column(f'csv/optim_{optims[2]}_activation{activation}_lr_0.01_lr_min1e-05_l2_0.0001.csv',3)
    d4 = read_csv_column(f'csv/optim_{optims[3]}_activation{activation}_lr_0.01_lr_min1e-05_l2_0.0001.csv',3)
    d5 = read_csv_column(f'csv/optim_{optims[4]}_activation{activation}_lr_0.01_lr_min1e-05_l2_0.0001.csv',3)
    d6 = read_csv_column(f'csv/optim_{optims[5]}_activation{activation}_lr_0.01_lr_min1e-05_l2_0.0001.csv',3)
    d7 = read_csv_column(f'csv/optim_{optims[6]}_activation{activation}_lr_0.01_lr_min1e-05_l2_0.0001.csv',3)
    d8 = read_csv_column(f'csv/optim_{optims[7]}_activation{activation}_lr_0.01_lr_min1e-05_l2_0.0001.csv',3)


    # 设置字体为Times New Roman，同时设置字体大小
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 16  # 将字体大小设置为16

    plt.figure(figsize=(10, 6), dpi=200)

    # 绘制测试损失曲线（加粗，橙色）
    plt.plot(range(1, 101), d1, color='black', label=f'{optims[0]}')
    plt.plot(range(1, 101), d2, color='red', label=f'{optims[1]}')
    plt.plot(range(1, 101), d3, color='peru', label=f'{optims[2]}')
    plt.plot(range(1, 101), d4, color='gold', label=f'{optims[3]}')
    plt.plot(range(1, 101), d5, color='lawngreen', label=f'{optims[4]}')
    plt.plot(range(1, 101), d6, color='cyan',label=f'{optims[5]}')
    plt.plot(range(1, 101), d7, color='blue',  label=f'{optims[6]}')
    plt.plot(range(1, 101), d8, color='purple', label=f'{optims[7]}')

    plt.xlabel('Number of Epochs', fontsize=20)  # 将x轴标签的字体大小设置为20
    plt.ylabel('Test Accuracy', fontsize=20)  # 将y轴标签的字体大小设置为20
    plt.xticks(fontsize=18)  # 将x轴刻度的字体大小设置为18
    plt.yticks(fontsize=18)  # 将y轴刻度的字体大小设置为18
    plt.legend(fontsize=18)  # 将图例的字体大小设置为18

    # 设置坐标轴和线宽度
    ax = plt.gca()
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)
    ax.spines['top'].set_linewidth(2)

    # 设置所有线条加粗
    for line in ax.lines:
        line.set_linewidth(2)

    plt.legend(fontsize=18, bbox_to_anchor=(1, 1), loc='upper left')


    plt.tight_layout()

    plt.savefig(f'optim_{activation}_Accuracy.png', format='png')

    plt.show()



