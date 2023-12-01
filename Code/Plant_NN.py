import numpy as np
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F

class Flatten(nn.Module):
    def forward(self, x):
        return self.flatten(x)
    
    def flatten(self, x):
        N = x.shape[0]
        return x.view(N, -1)

class Plant_NN(nn.Module):
    
    def __init__(self):
        super(Plant_NN, self).__init__()
        
        self.conv_layers = []
        self.fc_layers = []
        



    def add_conv_layer(self, cnn_input_channel, cnn_output_channel, cnn_padding, cnn_filter_size, cnn_stride, max_pool_size,
                       max_pool_stride, max_pool_padding, bn_on=True, dropout_on=True, dropout_p=0.5):
        cnn = nn.Conv2d(cnn_input_channel, cnn_output_channel, kernel_size=cnn_filter_size, padding=cnn_padding, stride=cnn_stride)
        
        bn = nn.BatchNorm2d(cnn_output_channel)
        
        relu = nn.ReLU()
        
        do = nn.Dropout2d(p=dropout_p)
        
        maxPool = nn.MaxPool2d(kernel_size=max_pool_size, padding=max_pool_padding, stride= max_pool_stride)
        
        if dropout_on:
            if bn_on:
                conv = nn.Sequential(cnn, bn, relu,do, maxPool)
            else:
                conv = nn.Sequential(cnn, relu, do, maxPool)
        else:
            if bn_on:
                conv = nn.Sequential(cnn, bn, relu, maxPool)
            else:
                conv = nn.Sequential(cnn, relu, maxPool)
                
        self.conv_layers.append(conv)
        
        
        
    def add_fc_layer(self, input_size, output_size, activation="relu", dropout_on=True, dropout_p=0.5):

        linear = nn.Linear(input_size, output_size)
        bn = nn.BatchNorm1d(output_size)
        
        if activation == "relu":
            acf = nn.ReLU(inplace=True)
        elif activation == "Tanh":
            acf = nn.Tanh()
        else:
            acf = nn.ReLU()
        if dropout_on:
            do = nn.Dropout(p=dropout_p)
            fc = nn.Sequential(do, linear, acf, bn)
        else:
            fc = nn.Sequential(linear, acf, bn)
        
        self.fc_layers.append(fc)
        
        
    def get_model(self):
        elements = []
        for cnv in self.conv_layers:
            elements.append(cnv)
        elements.append(Flatten())
        
        for fc in self.fc_layers:
            elements.append(fc)
            
        return nn.Sequential(*elements)
            