import torch
from torch import nn
#
from NeuronModel1 import Model
from torchsummary import summary
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#
# model_path = 'model/best_optim_adam_activation_ReLU_lr_0.01_lr_min_1e-05_l2_0.0001.pth'
# onnx_model_path = 'model.onnx'
#
# model = Model(nn.ReLU).to(device)
# model.load_state_dict(torch.load(model_path))
# model.eval()
# input = torch.randn(1, 3, 32, 32).to(device)
# torch.onnx.export(model,input, onnx_model_path, verbose=False)



model = Model(nn.ReLU).to(device)
summary(model,input_size=(3,32,32))
