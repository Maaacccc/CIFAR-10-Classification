import os
import torch
from requests import models
from sklearn.model_selection import ParameterGrid
from tensorboardX import SummaryWriter
from torch import optim, nn
from sklearn.metrics import classification_report,confusion_matrix
import torchvision.models as models
from Function import getDataLoader, train1
from NeuronModel1 import Model
from torchsummary import summary

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
batch_size = 32
trainloader, testloader = getDataLoader(batchsize=batch_size)

param_grid = {
    'l2_lambda': [1e-4],
    'lr': [1e-2],
    'lr_min': [1e-5],
    'optim':['adam'],
    'activation': [nn.ReLU],
    'num_epochs': [100]
}

for params in ParameterGrid(param_grid):

    optim = params['optim']
    l2_lambda = params['l2_lambda']
    lr = params['lr']
    lr_min = params['lr_min']
    activation = params['activation']
    num_epochs = params['num_epochs']


    log_dir = './logs'
    os.makedirs(log_dir, exist_ok=True)
    log_path = log_dir + '/'
    writer = SummaryWriter(os.path.join(log_path,f"neuron1"))

    model = Model(activation)

    criterion = nn.CrossEntropyLoss()
    train1(model, criterion, trainloader, testloader, device, batch_size=batch_size, num_epoch=num_epochs, lr=lr, lr_min=lr_min, l2_lambda=l2_lambda,optim=optim, init=False,writer=writer, activation=activation.__name__)
    writer.close()




