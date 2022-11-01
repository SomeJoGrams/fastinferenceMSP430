##!/usr/bin/env python3

import sys
import pandas as pd
import numpy as np
import os
import argparse

import json
import torch.onnx
import torch
from torch import nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torch.utils.data import TensorDataset
from torch.autograd import Function

curStep = 0


from test_utils import test_implementations
#from fastinference.Helper import NumpyEncoder

def sanatize_onnx(model):
    """ONNX does not support binary layers out of the box and exporting custom layers is sometimes difficult. This function sanatizes a given MLP so that it can be exported into an onnx file. To do so, it replaces all BinaryLinear layer with regular nn.Linear layers and BinaryTanh with Sign() layers. Weights and biases are copied and binarized as required.

    Args:
        model: The pytorch model.

    Returns:
        Model: The pytorch model in which each binary layer is replaced with the appropriate float layer.
    """

    # Usually I would use https://pytorch.org/docs/stable/generated/torch.heaviside.html for exporting here, but this is not yet supported in ONNX files. 
    class Sign(nn.Module):
        def forward(self, input):
            return torch.where(input > 0, torch.tensor([1.0]), torch.tensor([-1.0]))
            # return torch.sign(input)

    for name, m in model._modules.items():
        print("Checking {}".format(name))

        if isinstance(m, BinaryLinear):
            print("Replacing {}".format(name))
            # layer_old = m
            layer_new = nn.Linear(m.in_features, m.out_features, hasattr(m, 'bias'))
            if (hasattr(m, 'bias')):
                if (m.bias != None):
                    layer_new.bias.data = binarize(m.bias.data)
            layer_new.weight.data = binarize(m.weight.data)
            model._modules[name] = layer_new

        if isinstance(m, BinaryTanh):
            model._modules[name] = Sign()

        if isinstance(m, BinaryConv2d):
            print("Replacing {}".format(name))
            # layer_old = m
            layer_new = nn.Conv2d(
                in_channels = m.in_channels, 
                out_channels = m.out_channels, 
                kernel_size = m.kernel_size, 
                stride = m.stride, 
                #padding = m.padding,
                bias = hasattr(m, 'bias')
            )

            if (hasattr(m, 'bias')):
                layer_new.bias.data = binarize(m.bias.data)
            layer_new.weight.data = binarize(m.weight.data)
            model._modules[name] = layer_new
        
        # if isinstance(m, nn.BatchNorm2d):
        #     layer_new = WrappedBatchNorm(m)
        #     model._modules[name] = layer_new

    return model

class BinarizeF(Function):
    @staticmethod
    def forward(ctx, input):
        output = input.new(input.size())
        output[input > 0] = 1
        output[input <= 0] = -1
        return output

    @staticmethod
    def backward(ctx, grad_output):
        #return grad_output, None
        grad_input = grad_output.clone()
        return grad_input#, None

# aliases
binarize = BinarizeF.apply

class BinaryConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super(BinaryConv2d, self).__init__(*args, **kwargs)

    def forward(self, input):
        if self.bias is None:
            binary_weight = binarize(self.weight)
            return F.linear(input, binary_weight)
        else:            
            binary_weight = binarize(self.weight)
            binary_bias = binarize(self.bias)
            return F.conv2d(input, binary_weight, binary_bias)

class BinaryLinear(nn.Linear):
    def __init__(self, *args, **kwargs):
        super(BinaryLinear, self).__init__(*args, **kwargs)

    def forward(self, input):
        if self.bias is None:
            binary_weight = binarize(self.weight)
            # print("using the forward bin")
            return F.linear(input, binary_weight)
        else:
            binary_weight = binarize(self.weight)
            binary_bias = binarize(self.bias)

            return F.linear(input, binary_weight, binary_bias)

class BinaryTanh(nn.Module):
    def __init__(self, *args, **kwargs):
        super(BinaryTanh, self).__init__()
        self.hardtanh = nn.Hardtanh(*args, **kwargs)

    def forward(self, input):
        output = self.hardtanh(input)
        output = binarize(output)
        return output

class SimpleCNN(pl.LightningModule):

    def __init__(self, input_dim, n_classes, binarize = False, outpath = "."):
        super().__init__()
        # mnist images are (1, 28, 28) (channels, width, height) 
        if binarize:
            
            self.conv1 = BinaryConv2d(1, 32, 3, 1)
            self.bn_1 = nn.BatchNorm2d(32)
            self.activation_1 = BinaryTanh()
            self.pool_1 = nn.MaxPool2d(2)

            self.conv2 = BinaryConv2d(32, 32, 3, 1)
            self.bn_2 = nn.BatchNorm2d(32)
            self.activation_2 = BinaryTanh()
            self.pool_2 = nn.MaxPool2d(2)

            self.fc_1 = BinaryLinear(32 * 5 * 5, 32)
            self.bn = nn.BatchNorm1d(32)
            self.activation = BinaryTanh()
            self.out = BinaryLinear(32, 10)
        else:
            self.conv1 = nn.Conv2d(1, 32, 3, 1)
            self.bn_1 = nn.BatchNorm2d(32)
            self.activation_1 = nn.ReLU()
            self.pool_1 = nn.MaxPool2d(2)

            self.conv2 = nn.Conv2d(32, 32, 3, 1)
            self.bn_2 = nn.BatchNorm2d(32)
            self.activation_2 = nn.ReLU()
            self.pool_2 = nn.MaxPool2d(2)

            self.fc_1 = torch.nn.Linear(32 * 5 * 5, 32)
            self.bn = nn.BatchNorm1d(32)
            self.activation = nn.ReLU()
            self.out = torch.nn.Linear(32, 10)
        self.outpath = outpath
        self.input_dim = input_dim
        self.n_classes = n_classes
        
    def forward(self, x):
        batch_size = x.shape[0]
        # print("before view", x.size())
        # print(x[0:100])

        x = x.view((batch_size, 1, 28, 28)) # das fromt die 784 bilder pixel in 1 * 28 * 28 Pixel -> für meine Zwecke einfach beibehalten

        x = self.conv1(x)
        x = self.bn_1(x)
        x = self.activation_1(x)
        x = self.pool_1(x)

        x = self.conv2(x)
        x = self.bn_2(x)
        x = self.activation_2(x)
        x = self.pool_2(x)

        x = x.view(batch_size, -1)
        x = self.fc_1(x)
        x = self.bn(x)
        x = self.activation(x)
        x = self.out(x)
        x = torch.log_softmax(x, dim=1)

        return x

    def cross_entropy_loss(self, logits, labels):
        return F.nll_loss(logits, labels)
        #return None

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)
        if loss is not None:
            self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)
        if loss is not None:
            self.log('val_loss', loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def predict(self, X):
        return self.forward(torch.from_numpy(X).float()).argmax(axis=1)   
        
    def on_epoch_start(self):
        print('\n')

    def fit(self, X, y):
        XTrainT = torch.from_numpy(X).float()
        YTrainT = torch.from_numpy(y).long()

        train_dataloader = DataLoader(TensorDataset(XTrainT, YTrainT), batch_size=64)
        val_loader = None 

        trainer = pl.Trainer(max_epochs = 1, default_root_dir = self.outpath) #, progress_bar_refresh_rate = 0) # commented refresh rate, as not supported anymore
        trainer.fit(self, train_dataloader, val_loader)
        self.eval()

    def store(self, out_path, accuracy, model_name):
        dummy_x = torch.randn(1, self.input_dim, requires_grad=False)

        djson = {
            "accuracy":accuracy,
            "name":model_name
        }

        with open(os.path.join(out_path, model_name + ".json"), "w") as outfile:  
            json.dump(djson, outfile) #, cls=NumpyEncoder

        onnx_path = os.path.join(out_path,model_name+".onnx")
        print("Exporting {} to {}".format(model_name,onnx_path))
        model = sanatize_onnx(self)
        # https://github.com/pytorch/pytorch/issues/49229
        # set torch.onnx.TrainingMode.PRESERVE
        torch.onnx.export(model, dummy_x,onnx_path,training=torch.onnx.TrainingMode.PRESERVE,export_params=True,opset_version=11, do_constant_folding=True, input_names = ['input'],  output_names = ['output'], dynamic_axes={'input' : {0 : 'batch_size'},'output' : {0 : 'batch_size'}})
        
        return onnx_path
class SimpleBNN(pl.LightningModule):

    def __init__(self, input_dim, n_classes,binarize=True,outpath = "."):
        super().__init__()
        # mnist images are (1, 28, 28) (channels, width, height) 
        self.fc_0 = BinaryLinear(28*28,3)
        self.bn_0 = nn.BatchNorm1d(3)
        self.activation_1 = BinaryTanh()
        # self.pool_1 = nn.MaxPool2d(2)
        self.fc_1 = BinaryLinear(3, 32)
        self.bn_1 = nn.BatchNorm1d(32)
        self.activation = BinaryTanh()
        self.out = BinaryLinear(32, 10)

        
        self.outpath = outpath
        self.input_dim = input_dim
        self.n_classes = n_classes
        
    def forward(self, x):
        batch_size = x.shape[0]
        # x = x.view((batch_size, 1, 28, 28)) # das fromt die 784 bilder pixel in 1 * 28 * 28 Pixel -> für meine Zwecke einfach beibehalten

        # bilder runterskalieren um das netwerk kleiner zu bekommen

        x = x.view((batch_size,784)) # das fromt die 784 bilder pixel in 1 * 28 * 28 Pixel -> für meine Zwecke einfach beibehalten
        x = self.fc_0(x)
        x = self.bn_0(x)
        x = self.activation_1(x)
        # x = self.pool_1(x)

        x = x.view(batch_size, -1)
        x = self.fc_1(x)
        x = self.bn_1(x)
        x = self.activation(x)
        x = self.out(x)
        x = torch.log_softmax(x, dim=1)

        # x = self.bn_2(x)

        return x

    def cross_entropy_loss(self, logits, labels):
        return F.nll_loss(logits, labels)
        #return None

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)
        if loss is not None:
            self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)
        if loss is not None:
            self.log('val_loss', loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def predict(self, X):
        return self.forward(torch.from_numpy(X).float()).argmax(axis=1)   
        
    def on_epoch_start(self):
        print('\n')

    def fit(self, X, y):
        XTrainT = torch.from_numpy(X).float()
        YTrainT = torch.from_numpy(y).long()

        train_dataloader = DataLoader(TensorDataset(XTrainT, YTrainT), batch_size=64)
        val_loader = None 

        trainer = pl.Trainer(max_epochs = 1, default_root_dir = self.outpath) #, progress_bar_refresh_rate = 0) # commented refresh rate, as not supported anymore
        trainer.fit(self, train_dataloader, val_loader)
        self.eval()

    def store(self, out_path, accuracy, model_name):
        dummy_x = torch.randn(1, self.input_dim, requires_grad=False)

        djson = {
            "accuracy":accuracy,
            "name":model_name
        }

        with open(os.path.join(out_path, model_name + ".json"), "w") as outfile:  
            json.dump(djson, outfile) #, cls=NumpyEncoder

        onnx_path = os.path.join(out_path,model_name+".onnx")
        print("Exporting {} to {}".format(model_name,onnx_path))
        model = sanatize_onnx(self)
        # https://github.com/pytorch/pytorch/issues/49229
        # set torch.onnx.TrainingMode.PRESERVE
        torch.onnx.export(model, dummy_x,onnx_path,training=torch.onnx.TrainingMode.PRESERVE,export_params=True,opset_version=11, do_constant_folding=True, input_names = ['input'],  output_names = ['output'], dynamic_axes={'input' : {0 : 'batch_size'},'output' : {0 : 'batch_size'}})
        
        return onnx_path

shortConst = 5

class SimpleBNNForEEGEye(pl.LightningModule):
    def __init__(self, input_dim, n_classes,binarize=True,outpath = "."):
        super().__init__()

        self.fc_0 = BinaryLinear(14,32*shortConst) # 14 input # for this 8 bits should be supported? -> the original vlaues should be brought in 8 bit form
        #self.fc_0 = BinaryLinear(32,32) # 14 input # for this 8 bits should be supported? -> the original vlaues should be brought in 8 bit form
        # currently expected are 14 32 bit values as input
        self.bn_0 = nn.BatchNorm1d(32 * shortConst)  # 32 Werte als output -> 
                                            # diese werden für das nächste layer auf 1 wert aus 32 bit zusammengefasst
                                            # jeder dieser 32 werte wird gebatchnormt ah, das ist falsch oder??
        self.activation_1 = BinaryTanh()
        # self.pool_1 = nn.MaxPool2d(2)
        # self.fc_1 = BinaryLinear(32, 16)
        # self.bn_1 = nn.BatchNorm1d(16)
        # self.activation = BinaryTanh()
        self.fc_1 = BinaryLinear(32*shortConst, 1)
        self.float()
        
        self.outpath = outpath
        self.input_dim = input_dim
        self.n_classes = n_classes
        
    def forward(self, x):
        # fill the rest with zeros?

        batch_size = x.shape[0]
        # x = x.view((batch_size, 1, 28, 28)) # das fromt die 784 bilder pixel in 1 * 28 * 28 Pixel -> für meine Zwecke einfach beibehalten
        # bilder runterskalieren um das netwerk kleiner zu bekommen

        x = x.view((batch_size,14)) # das fromt die 784 bilder pixel in 1 * 28 * 28 Pixel -> für meine Zwecke einfach beibehalten
        x = self.fc_0(x)
        # print("the forward result",x)
        # sys.exit()

        x = self.bn_0(x)
        x = self.activation_1(x)
        # x = self.pool_1(x)

        # x = x.view(batch_size, -1)
        # x = self.fc_1(x)
        # x = self.bn_1(x)
        # x = self.activation(x)
        x = self.fc_1(x)
        return x

    # def cross_entropy_loss(self, logits, labels):
    #     return F.nll_loss(logits, labels)
    #     #return None

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        logits = self.forward(x)
        #loss = F.l1_loss(logits, y)
        # l1_loss vs mse_loss
        loss = F.l1_loss(logits, y.float()) # convert to float to get a correct value
        if loss is not None:
            self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        logits = self.forward(x)
        loss = F.l1_loss(logits, y.float()) # convert to float to get a correct value

        # loss = F.l1_loss(logits, y)
        if loss is not None:
            self.log('val_loss', loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def predict(self, X):
        # result = self.forward(torch.from_numpy(X).detach().float())
        # result = self.forward(torch.from_numpy(X).detach().float())
        result = self.forward(torch.from_numpy(X).detach().float())
        resultFixed = torch.round(torch.clamp(result,0,1),decimals = 0)

        print("prediction result", resultFixed) # here all are zero
        return resultFixed
        
    def on_epoch_start(self):
        print('\n')

    def fit(self, X, y):
        XTrainT = torch.from_numpy(X).float()
        YTrainT = torch.from_numpy(y).long()

        train_dataloader = DataLoader(TensorDataset(XTrainT, YTrainT), batch_size=64)
        val_loader = None 

        trainer = pl.Trainer(max_epochs = 10, default_root_dir = self.outpath) #, progress_bar_refresh_rate = 0) # commented refresh rate, as not supported anymore
        trainer.fit(self, train_dataloader, val_loader)
        self.eval()

    def store(self, out_path, accuracy, model_name):
        dummy_x = torch.randn(1, self.input_dim, requires_grad=False)

        djson = {
            "accuracy":accuracy,
            "name":model_name
        }

        with open(os.path.join(out_path, model_name + ".json"), "w") as outfile:  
            json.dump(djson, outfile) #, cls=NumpyEncoder

        onnx_path = os.path.join(out_path,model_name+".onnx")
        print("Exporting {} to {}".format(model_name,onnx_path))
        model = sanatize_onnx(self)
        # https://github.com/pytorch/pytorch/issues/49229
        # set torch.onnx.TrainingMode.PRESERVE
        torch.onnx.export(model, dummy_x,onnx_path,training=torch.onnx.TrainingMode.PRESERVE,export_params=True,opset_version=11, do_constant_folding=True, input_names = ['input'],  output_names = ['output'], dynamic_axes={'input' : {0 : 'batch_size'},'output' : {0 : 'batch_size'}})
        
        return onnx_path



class SimpleNormalNNForEEGEye(pl.LightningModule):
    def __init__(self, input_dim, n_classes,binarize=True,outpath = "."):
        super().__init__()

        self.fc_0 = nn.Linear(14,30) # 14 input # for this 8 bits should be supported? -> the original vlaues should be brought in 8 bit form
        self.bn_0 = nn.BatchNorm1d(30)  # 32 Werte als output -> 
                                            # diese werden für das nächste layer auf 1 wert aus 32 bit zusammengefasst
                                            # jeder dieser 32 werte wird gebatchnormt ah, das ist falsch oder??
        #self.bn_0 = nn.Dropout()
        self.activation_1 = nn.ReLU()
        self.fc_1 = nn.Linear(30, 100)
        # self.bn_1 = nn.Dropout()
        self.bn_1 = nn.BatchNorm1d(100)
        self.activation_2 = nn.ReLU()
        self.fc_2 = nn.Linear(100, 10)
        # self.bn_2 = nn.Dropout()
        self.bn_2 = nn.BatchNorm1d(10)
        self.activation_3 = nn.ReLU()
        self.fc_3 = nn.Linear(10, 1)
        # self.bn_3 = nn.Dropout()

        self.bn_3 = nn.BatchNorm1d(1)
        self.float()
        
        self.outpath = outpath
        self.input_dim = input_dim
        self.n_classes = n_classes
        
    def forward(self, x):
        # fill the rest with zeros?
        batch_size = x.shape[0]

        x = x.view((batch_size,14))
        x = self.fc_0(x)
        x = self.bn_0(x)
        x = self.activation_1(x)
        x = self.fc_1(x)
        x = self.bn_1(x)
        x = self.activation_2(x)
        x = self.fc_2(x)
        x = self.bn_2(x)
        x = self.activation_3(x)
        x = self.fc_3(x)
        # x = F.softmax(x,1)
        # torch.set_printoptions(profile="full")
        # print("before ",x)
        x = self.bn_3(x)

        # print("after", x)
        # torch.set_printoptions(profile="default") 
        # print("the out vector",x)
        return x

    # def cross_entropy_loss(self, logits, labels):
    #     return F.nll_loss(logits, labels)
    #     #return None

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        # transform every output so that we get 2 error classes for 1 and 0, how to train this otherwise with batch norm?
        # input 1x3  [0,1,1] , goal 2x3 [[0,1],[1,0],[1,0]]
        # transformedLabels = [] # transform labels to classes
        # for inp in y: # if the input is zero the second class is chosen, else the first
        #     if (inp == 0):
        #         transformedLabels.append((0,1)) # class 0 
        #     else: # inp == 1
        #         transformedLabels.append((1,0)) # class 1
        # transformedLabels = torch.tensor(transformedLabels,dtype=torch.float)
        # transformedLabels = torch.tensor(transformedLabels,dtype=torch.long)
        logits = self.forward(x)
        #loss = F.l1_loss(logits, y)
        # l1_loss vs mse_loss
        # print("x", x.shape,"y",y.shape, "types", type(x), type(y))
        # y = transformedLabels
        # y = y.reshape((y.shape[0],1))
        #loss = F.l1_loss(logits, y.float()) # convert to float to get a correct value
        # loss = F.mse_loss(logits, y.float(),reduction="none") # convert to float to get a correct value # Problem is the average is over all the outputs
        # print("\n")
        # print("labels", y)
        # print("input", x)
        # print("logits",logits)
        

        loss = F.mse_loss(logits, y) # convert to float to get a correct value # Problem is the average is over all the outputs
        # if (batch_idx % 100 == 0):
        #     print("\n\n")
        #     print("inp", logits)
        #     print("output", y.float())
        #     print("loss", loss)
        # print("calculated loss vector", loss)
        # print("input", logits, "\n\noutput", y.float())
        # print("the loss", loss)
        # if (batch_idx % 100 == 0):
        #     print("the calculated losses ,", loss)

        if loss is not None:
            self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        logits = self.forward(x)
        # loss = F.l1_loss(logits, y.float()) # convert to float to get a correct value
        loss = F.mse_loss(logits, y.float()) # convert to float to get a correct value

        if loss is not None:
            self.log('val_loss', loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def predict(self, X):
        # result = self.forward(torch.from_numpy(X).detach().float())
        # result = self.forward(torch.from_numpy(X).detach().float())
        forResult = self.forward(torch.from_numpy(X).detach().float())
        print(forResult)
        result = forResult.argmax(axis=1)
        # return the higher value of the two outputs
        # resultFixed = torch.round(torch.abs(result),decimals=0) #,0,1),decimals = 0)
        # resultFixed = []
        # print(result)
        # for out in result:
        #     zeroChance,oneChance = out[0],out[1]
        #     if (zeroChance > oneChance):
        #         resultFixed.append(0)
        #     else:
        #         resultFixed.append(1)

        print("prediction result", result) # here all are zero
        # return torch.tensor(result)
        return result
        
    def on_epoch_start(self):
        print('\n')

    def fit(self, X, y):
        XTrainT = torch.from_numpy(X).float()
        YTrainT = torch.from_numpy(y).float()

        train_dataloader = DataLoader(TensorDataset(XTrainT, YTrainT), batch_size=2)
        val_loader = None 

        trainer = pl.Trainer(max_epochs = 5, default_root_dir = self.outpath) #, progress_bar_refresh_rate = 0) # commented refresh rate, as not supported anymore
        trainer.fit(self, train_dataloader, val_loader)
        self.eval()

    def store(self, out_path, accuracy, model_name):
        dummy_x = torch.randn(1, self.input_dim, requires_grad=False)

        djson = {
            "accuracy":accuracy,
            "name":model_name
        }

        with open(os.path.join(out_path, model_name + ".json"), "w") as outfile:  
            json.dump(djson, outfile) #, cls=NumpyEncoder

        onnx_path = os.path.join(out_path,model_name+".onnx")
        print("Exporting {} to {}".format(model_name,onnx_path))
        model = sanatize_onnx(self)
        # https://github.com/pytorch/pytorch/issues/49229
        # set torch.onnx.TrainingMode.PRESERVE
        torch.onnx.export(model, dummy_x,onnx_path,training=torch.onnx.TrainingMode.PRESERVE,export_params=True,opset_version=11, do_constant_folding=True, input_names = ['input'],  output_names = ['output'], dynamic_axes={'input' : {0 : 'batch_size'},'output' : {0 : 'batch_size'}})
        
        return onnx_path
class SimpleNNForEEGEye(pl.LightningModule):
    def __init__(self, input_dim, n_classes,binarize=True,outpath = "."):
        super().__init__()

        self.fc_0 = nn.Linear(14,200) # 14 input # for this 8 bits should be supported? -> the original vlaues should be brought in 8 bit form
        # self.bn_0 = nn.Dropout()
        self.bn_0 = nn.BatchNorm1d(200)
        self.activation_1 = nn.ReLU()
        self.fc_1 = nn.Linear(200,200)
        # self.bn_1 = nn.Dropout()
        self.bn_1 = nn.BatchNorm1d(200)
        
        self.activation_2 = nn.ReLU()
        self.fc_2 = nn.Linear(200,2)
        self.bn_2 = nn.BatchNorm1d(2)

        
      
        self.outpath = outpath
        self.input_dim = input_dim
        self.n_classes = n_classes
        
    def forward(self, x):
        # fill the rest with zeros?
        batch_size = x.shape[0]

        x = x.view((batch_size,14))
        x = self.fc_0(x)
        x = self.bn_0(x)
        x = self.activation_1(x)
        x = self.fc_1(x)
        x = self.bn_1(x)
        x = self.activation_2(x)
        x = self.fc_2(x)
        # x = self.bn_2(x)
        # x = self.bn_0(x)
        # x = self.activation_1(x)
        return x

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        # transform every output so that we get 2 error classes for 1 and 0, how to train this otherwise with batch norm?
        # input 1x3  [0,1,1] , goal 2x3 [[0,1],[1,0],[1,0]]
        # transformedLabels = [] # transform labels to classes
        # transformedLabels = torch.tensor(transformedLabels,dtype=torch.float)
        # transformedLabels = torch.tensor(transformedLabels,dtype=torch.long)
        global curStep

        logits = self.forward(x)

        #loss = F.l1_loss(logits, y)
        # l1_loss vs mse_loss
        #loss = F.l1_loss(logits, y.float()) # convert to float to get a correct value
        # loss = F.mse_loss(logits, y.float(),reduction="none") # convert to float to get a correct value # Problem is the average is over all the outputs
        # transform y into a classification problem
        changedY = []
        for nr in y:
            if (nr == 0):
                changedY.append(0.)
            else:
                changedY.append(1.) # meaning label1 is the correct label
        # for nr in y:
        #     if (nr == 0):
        #         changedY.append(0.)
        #     else:
        #         changedY.append(1.) # meaning label1 is the correct label
        changedY = torch.tensor(changedY,dtype=torch.long)

        lossFunc = torch.nn.CrossEntropyLoss()
        # loss = F.cross_entropy(logits, changedY) # convert to float to get a correct value # Problem is the average is over all the outputs
        loss = lossFunc(logits, changedY) # convert to float to get a correct value # Problem is the average is over all the outputs

        curStep += 1

        if loss is not None:
            self.log('train_loss', loss)
        return loss

    # def validation_step(self, val_batch, batch_idx):
        # x, y = val_batch
        # logits = self.forward(x)
        # # loss = F.l1_loss(logits, y.float()) # convert to float to get a correct value
        # loss = F.mse_loss(logits, y.float()) # convert to float to get a correct value

        # if loss is not None:
        #     self.log('val_loss', loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def predict(self, X):
        # result = self.forward(torch.from_numpy(X).detach().float())
        # result = self.forward(torch.from_numpy(X).detach().float())
        forResult = self.forward(torch.from_numpy(X).detach().float()).argmax(axis=1)
        print("the predict result", forResult) # the result closer to 0
        
        result = forResult
        # result = forResult.argmax(axis=1)
        # return the higher value of the two outputs
        # resultFixed = torch.round(torch.abs(result),decimals=0) #,0,1),decimals = 0)
        # resultFixed = []
        # print(result)
        # for out in result:
        #     zeroChance,oneChance = out[0],out[1]
        #     if (zeroChance > oneChance):
        #         resultFixed.append(0)
        #     else:
        #         resultFixed.append(1)
        # return torch.tensor(result)
        return result
        
    def on_epoch_start(self):
        print('\n')

    def fit(self, X, y):
        XTrainT = torch.from_numpy(X).float()
        YTrainT = torch.from_numpy(y).float()

        train_dataloader = DataLoader(TensorDataset(XTrainT, YTrainT), batch_size=512)
        val_loader = None 

        trainer = pl.Trainer(max_epochs = 40, default_root_dir = self.outpath) #, progress_bar_refresh_rate = 0) # commented refresh rate, as not supported anymore
        trainer.fit(self, train_dataloader, val_loader)
        self.eval()

    def store(self, out_path, accuracy, model_name):
        dummy_x = torch.randn(1, self.input_dim, requires_grad=False)

        djson = {
            "accuracy":accuracy,
            "name":model_name
        }

        with open(os.path.join(out_path, model_name + ".json"), "w") as outfile:  
            json.dump(djson, outfile) #, cls=NumpyEncoder

        onnx_path = os.path.join(out_path,model_name+".onnx")
        print("Exporting {} to {}".format(model_name,onnx_path))
        model = sanatize_onnx(self)
        # https://github.com/pytorch/pytorch/issues/49229
        # set torch.onnx.TrainingMode.PRESERVE
        torch.onnx.export(model, dummy_x,onnx_path,training=torch.onnx.TrainingMode.PRESERVE,export_params=True,opset_version=11, do_constant_folding=True, input_names = ['input'],  output_names = ['output'], dynamic_axes={'input' : {0 : 'batch_size'},'output' : {0 : 'batch_size'}})
        
        return onnx_path
class DropoutNNForEEGEye(pl.LightningModule):
    def __init__(self, input_dim, n_classes,binarize=True,outpath = "."):
        super().__init__()

        self.fc_0 = nn.Linear(14,40) # 14 input # for this 8 bits should be supported? -> the original vlaues should be brought in 8 bit form
        self.activation_1 = nn.ReLU()
        self.bn_0 = nn.Dropout()
        self.fc_1 = nn.Linear(40,2)
        # self.outputActivation = nn.ReLU()

        # self.bn_2 = nn.BatchNorm1d(2)

        self.outpath = outpath
        self.input_dim = input_dim
        self.n_classes = n_classes
        
    def forward(self, x):
        batch_size = x.shape[0]

        x = x.view((batch_size,14))
        x = self.fc_0(x)
        x = self.activation_1(x)
        x = self.bn_0(x)
        x = self.fc_1(x)
        # x = self.outputActivation(x)
        # x = self.activation_2(x)
        # x = self.bn_1(x)
        # x = self.fc_2(x)
        # x = self.bn_2(x)
        return x

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        # transform every output so that we get 2 error classes for 1 and 0, how to train this otherwise with batch norm?
        # input 1x3  [0,1,1] , goal 2x3 [[0,1],[1,0],[1,0]]
        # transformedLabels = [] # transform labels to classes
        # transformedLabels = torch.tensor(transformedLabels,dtype=torch.float)
        # transformedLabels = torch.tensor(transformedLabels,dtype=torch.long)
        global curStep

        logits = self.forward(x)

        #loss = F.l1_loss(logits, y)
        # l1_loss vs mse_loss
        # print("x", x.shape,"y",y.shape, "types", type(x), type(y))
        # y = transformedLabels
        # y = y.reshape((y.shape[0],1))
        #loss = F.l1_loss(logits, y.float()) # convert to float to get a correct value
        # loss = F.mse_loss(logits, y.float(),reduction="none") # convert to float to get a correct value # Problem is the average is over all the outputs
        # transform y into a classification problem
        changedY = []
        # for nr in y:
        #     if (nr == 0):
        #         changedY.append([0.,1.])
        #     else:
        #         changedY.append([1.,0.]) # meaning label1 is the correct label
        # changedY = torch.tensor(changedY,dtype=torch.float)

        for nr in y:
            if (nr == 0): 
                changedY.append(0.) # eye closed label
            else:
                changedY.append(1.) # meaning label1 is the correct label
        changedY = torch.tensor(changedY,dtype=torch.long)

        loss = F.cross_entropy(logits, changedY,reduction="mean") # convert to float to get a correct value # Problem is the average is over all the outputs
        # print("current RUN")
        # print("input",logits)
        # print("output", changedY)
        # print("loss", loss)
        curStep += 1

        if loss is not None:
            self.log('train_loss', loss)
        return loss

    # def validation_step(self, val_batch, batch_idx):
        # x, y = val_batch
        # logits = self.forward(x)
        # # loss = F.l1_loss(logits, y.float()) # convert to float to get a correct value
        # loss = F.mse_loss(logits, y.float()) # convert to float to get a correct value

        # if loss is not None:
        #     self.log('val_loss', loss)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=0.1,momentum=0.9),#lr=1e-3)
        return optimizer

    def predict(self, X):
        # result = self.forward(torch.from_numpy(X).detach().float())
        # result = self.forward(torch.from_numpy(X).detach().float())

        forResult = self.forward(torch.from_numpy(X).detach().float())
        print("the for result")
        print(forResult)
        result = torch.softmax(forResult,dim=1).argmax(axis=1)
        print(result)
        # result = forResult.argmax(axis=1)
        # return the higher value of the two outputs
        # resultFixed = torch.round(torch.abs(result),decimals=0) #,0,1),decimals = 0)
        # resultFixed = []
        # print(result)
        # for out in result:
        #     zeroChance,oneChance = out[0],out[1]
        #     if (zeroChance > oneChance):
        #         resultFixed.append(0)
        #     else:
        #         resultFixed.append(1)

        # return torch.tensor(result)
        return result
        
    def on_epoch_start(self):
        print('\n')

    def fit(self, X, y):
        XTrainT = torch.from_numpy(X).float()
        YTrainT = torch.from_numpy(y).long()

        train_dataloader = DataLoader(TensorDataset(XTrainT, YTrainT), batch_size=1024)
        val_loader = None 

        trainer = pl.Trainer(max_epochs = 40, default_root_dir = self.outpath) #, progress_bar_refresh_rate = 0) # commented refresh rate, as not supported anymore
        trainer.fit(self, train_dataloader, val_loader)
        self.eval()

    def store(self, out_path, accuracy, model_name):
        dummy_x = torch.randn(1, self.input_dim, requires_grad=False)

        djson = {
            "accuracy":accuracy,
            "name":model_name
        }

        with open(os.path.join(out_path, model_name + ".json"), "w") as outfile:  
            json.dump(djson, outfile) #, cls=NumpyEncoder

        onnx_path = os.path.join(out_path,model_name+".onnx")
        print("Exporting {} to {}".format(model_name,onnx_path))
        model = sanatize_onnx(self)
        # https://github.com/pytorch/pytorch/issues/49229
        # set torch.onnx.TrainingMode.PRESERVE
        torch.onnx.export(model, dummy_x,onnx_path,training=torch.onnx.TrainingMode.PRESERVE,export_params=True,opset_version=11, do_constant_folding=True, input_names = ['input'],  output_names = ['output'], dynamic_axes={'input' : {0 : 'batch_size'},'output' : {0 : 'batch_size'}})
        
        return onnx_path
class BNBNNForEEGEye(pl.LightningModule):
    def __init__(self, input_dim, n_classes,binarize=True,outpath = "."):
        super().__init__()

        # self.fc_0 = BinaryLinear(14,32,bias=False) # 14 input # for this 8 bits should be supported? -> the original vlaues should be brought in 8 bit form
        # print("weight amount", len(self.fc_0.weight[0]), self.fc_0.weight[0])
        # sys.exit()
        self.bn_0 = nn.BatchNorm1d(14)
        self.activation_0 = BinaryTanh()
        self.fc_1 = BinaryLinear(14,32*5,bias=False)
        self.bn_1 = nn.BatchNorm1d(32*5) #-> this layer will concatenate to 1 value in 32 bit, 2 in 16 bit ...

        self.activation_1 = BinaryTanh()
        self.fc_2 = BinaryLinear(32*5,64,bias=False)
        self.bn_2 = nn.BatchNorm1d(64)
        self.activation_2 = BinaryTanh()
        self.fc_3 = BinaryLinear(64,2,bias=False)
        self.bn_3 = nn.BatchNorm1d(2)

        # self.outputActivation = nn.ReLU()
        self.outpath = outpath
        self.input_dim = input_dim
        self.n_classes = n_classes
        
    def forward(self, x,printForward=False):
        batch_size = x.shape[0]
        x = x.view((batch_size,14))
        if (printForward):
            print("bn input")
            print(x)
        x = self.bn_0(x)
        if (printForward):
            print("the bn0 layer")
            print("bn0 normalisation", self.bn_0,"\n")
            print("bn0 weights", self.bn_0.weight, "\n")
            print("bn0 bias", self.bn_0.bias, "\n")

            print("tensor test 123")
            print("bn0",x)
        x = self.activation_0(x)
        if (printForward):
            print("act0",x)
        x = self.fc_1(x)
        if (printForward):
            print("fc1 result",x)
            print("fully connected weights", self.fc_1.weight)
            print("fc1 last", x[len(x) - 1])
            print("fc1 prelast", x[len(x) - 2])
        x = self.bn_1(x)
        if (printForward):
            print("the bn1 layer")
            print("bn1 normalisation", self.bn_1,"\n")
            print("bn1 weights", self.bn_1.weight, "\n")
            print("bn1 bias", self.bn_1.bias, "\n")
        x = self.activation_1(x)
        if (printForward):
            print("act1",x)
            print(x[len(x) - 1])
            print("prelast", x[len(x) - 2])
            print("\n") # maybe also use full print
        x = self.fc_2(x)
        if (printForward):
            print("fc2",x)
        x = self.bn_2(x)
        if (printForward):
            print("the bn_2 layer")
            print("the bn2 layer")
            print("bn2 normalisation", self.bn_2,"\n")
            print("bn2 weights", self.bn_2.weight, "\n")
            print("bn2 bias", self.bn_2.bias, "\n")
            print("result",x)
        x = self.activation_2(x)
        if (printForward):
            print("act2",x)
            print(x[len(x) - 1])
            print("\n") # maybe also use full print
        # x = self.bn_1(x)
        x = self.fc_3(x)
        if (printForward):
            print("result final fc", x)
        x = self.bn_3(x)
        if (printForward):
            print("result final batchnorm", x,"\n\n")
        return x

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        # transform every output so that we get 2 error classes for 1 and 0, how to train this otherwise with batch norm?
        # input 1x3  [0,1,1] , goal 2x3 [[0,1],[1,0],[1,0]]
        # transformedLabels = [] # transform labels to classes
        # transformedLabels = torch.tensor(transformedLabels,dtype=torch.float)
        # transformedLabels = torch.tensor(transformedLabels,dtype=torch.long)
        global curStep

        logits = self.forward(x)

        #loss = F.l1_loss(logits, y)
        # l1_loss vs mse_loss
        # print("x", x.shape,"y",y.shape, "types", type(x), type(y))
        # y = transformedLabels
        # y = y.reshape((y.shape[0],1))
        #loss = F.l1_loss(logits, y.float()) # convert to float to get a correct value
        # loss = F.mse_loss(logits, y.float(),reduction="none") # convert to float to get a correct value # Problem is the average is over all the outputs
        # transform y into a classification problem
        changedY = []
        # for nr in y:
        #     if (nr == 0):
        #         changedY.append([0.,1.])
        #     else:
        #         changedY.append([1.,0.]) # meaning label1 is the correct label
        # changedY = torch.tensor(changedY,dtype=torch.float)

        for nr in y:
            if (nr == 0): 
                # changedY.append(-1.) # eye closed label # hingeEmbedding loss uses -1 and 1 as labels
                changedY.append(int(0)) # eye closed label
            else:
                # changedY.append(1.) # meaning label1 is the correct label
                changedY.append(int(1)) # meaning label1 is the correct label
        changedY = torch.tensor(changedY,dtype=torch.long)

        loss = F.cross_entropy(logits, changedY,reduction="mean") # convert to float to get a correct value # Problem is the average is over all the outputs
        # lossFunc = torch.nn.HingeEmbeddingLoss()
        # loss = lossFunc(logits, changedY) # convert to float to get a correct value # Problem is the average is over all the outputs
        # print("current RUN")
        # print("input",logits)
        # print("output", changedY)
        # print("loss", loss)
        curStep += 1

        if loss is not None:
            self.log('train_loss', loss)
        return loss

    # def validation_step(self, val_batch, batch_idx):
        # x, y = val_batch
        # logits = self.forward(x)
        # # loss = F.l1_loss(logits, y.float()) # convert to float to get a correct value
        # loss = F.mse_loss(logits, y.float()) # convert to float to get a correct value

        # if loss is not None:
        #     self.log('val_loss', loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.1),#lr=1e-3)
        return optimizer

    def predict(self, X):
        # result = self.forward(torch.from_numpy(X).detach().float())
        # result = self.forward(torch.from_numpy(X).detach().float())

        forResult = self.forward(torch.from_numpy(X).detach().float(),)

        result = torch.softmax(forResult,dim=1)
        # print("predict result", result)
        # for ind,inp in enumerate(torch.from_numpy(X).detach().float()):
        #     curRes = self.forward(inp.reshape(1,-1))
        #     print(curRes,curRes.argmax(axis=1))
        #     if (ind == 100):
        #         sys.exit()
        # result = forResult.argmax(axis=1)
        # return the higher value of the two outputs
        # resultFixed = torch.round(torch.abs(result),decimals=0) #,0,1),decimals = 0)
        # resultFixed = []
        # print(result)
        # for out in result:
        #     zeroChance,oneChance = out[0],out[1]
        #     if (zeroChance > oneChance):
        #         resultFixed.append(0)
        #     else:
        #         resultFixed.append(1)

        # return torch.tensor(result)
        return result.argmax(axis=1)
        
    def predictLogits(self,X,printForward = False):
        forResult = self.forward(torch.from_numpy(X).detach().float(),printForward=printForward)
        return forResult

    def on_epoch_start(self):
        print('\n')

    def fit(self, X, y):
        XTrainT = torch.from_numpy(X).float()
        YTrainT = torch.from_numpy(y).long()

        train_dataloader = DataLoader(TensorDataset(XTrainT, YTrainT), batch_size=256,shuffle=True)
        val_loader = None 

        trainer = pl.Trainer(max_epochs = 10,default_root_dir = self.outpath) #, progress_bar_refresh_rate = 0) # commented refresh rate, as not supported anymore
        trainer.fit(self, train_dataloader, val_loader)
        self.eval()

    def store(self, out_path, accuracy, model_name):
        dummy_x = torch.randn(1, self.input_dim, requires_grad=False)

        djson = {
            "accuracy":accuracy,
            "name":model_name
        }

        with open(os.path.join(out_path, model_name + ".json"), "w") as outfile:  
            json.dump(djson, outfile) #, cls=NumpyEncoder

        onnx_path = os.path.join(out_path,model_name+".onnx")
        print("Exporting {} to {}".format(model_name,onnx_path))
        model = sanatize_onnx(self)
        # https://github.com/pytorch/pytorch/issues/49229
        # set torch.onnx.TrainingMode.PRESERVE
        torch.onnx.export(model, dummy_x,onnx_path,training=torch.onnx.TrainingMode.PRESERVE,export_params=True,opset_version=11, do_constant_folding=True, input_names = ['input'],  output_names = ['output'], dynamic_axes={'input' : {0 : 'batch_size'},'output' : {0 : 'batch_size'}})
        
        return onnx_path

def main():
    parser = argparse.ArgumentParser(description='Benchmark various CNN optimizations on the MNIST / Fashion dataset.')
    scriptPath = os.path.dirname(os.path.abspath(__file__))
    defaultPath = os.path.join(scriptPath,"generatedModels")
    parser.add_argument('--outpath', required=False,default=defaultPath, help='Folder where data should written to.')
    parser.add_argument('--modelname', required=False, default="model", help='Modelname')
    parser.add_argument('--split','-s', required=False, default=0.1, type=float, help='Test/Train split.')
    parser.add_argument('--dataset','-d', required=False,default="eeg", help='Dataset to to be downloaded and used. Currently supported are {mnist, fashion}.')
    parser.add_argument("--binarize", "-b", required=False, action='store_true', help="Trains a binarized neural network if true.")
    
    
    parser.add_argument("--createNewModel", "-nm", required=False,action='store_true', help="Creates a new model if true, otherwise tries to load a model.")
    # store_true makes the default to false -> default case is no newModelCreation
    
    args = parser.parse_args()



    model = None
    trainedModelPath = None

    modelName = args.modelname + args.dataset

    if (args.createNewModel): 
        try:
            os.remove(os.path.join(defaultPath,modelName +  ".onnx"))
        except FileNotFoundError:
            print("onnx Model File didnt exist")
        try:
            os.remove(os.path.join(defaultPath,modelName + ".json"))
        except FileNotFoundError:
            print("json Acc File didnt exist")
        

    if args.dataset in ["mnist","fashion"]:
        n_features, n_classes = 28*28,10

        # model = SimpleCNN(n_features, n_classes, args.binarize, args.outpath)
        model = SimpleCNN(n_features, n_classes, True, args.outpath)

    elif args.dataset in ["eeg"]:
        n_features, n_classes = 14, 2
        storedModelPath = os.path.join(defaultPath,modelName)
        if (os.path.exists(storedModelPath + ".onnx") and os.path.exists(storedModelPath + ".json")):
            model = None #load the model from path through fastinference if it exists as onnx model
            trainedModelPath = storedModelPath # this is only the model name without fileextension
        else:
            model = BNBNNForEEGEye(n_features,n_classes,args.binarize, args.outpath)
    # elif args.dataset in []:
    else:
        print("Only {eeg, magic, mnist, fashion} is supported for the --dataset/-d argument but you supplied {}.".format(args.dataset))
        sys.exit(1)


    args.binarize = True # use the binary Layers

    # implementations = [ 
    #     ("NHWC",{}) 
    # ]

    implementations = [ 
        ("MSP430",{}) 
    ]

    if args.binarize:
        implementations.append( ("binary",{}) )

    optimizers = [
        ([None], [{}])
    ]

    performance = test_implementations(model = model, dataset= args.dataset, split = args.split, implementations = implementations
        , base_optimizers = optimizers, out_path = args.outpath, model_name = modelName,trainedModelPath=trainedModelPath)
    df = pd.DataFrame(performance)
    print(df)

if __name__ == '__main__':
    main()