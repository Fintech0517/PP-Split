'''
Author: Ruijun Deng
Date: 2023-08-27 20:58:31
LastEditTime: 2023-12-12 21:06:17
LastEditors: Ruijun Deng
FilePath: /PP-Split/target_model/models/VGG.py
Description: 
'''

import time
import math
import os
import sys
import numpy as np

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

import collections


# --------- VGG model ---------
# split_layer_list = list(range(len(model_cfg['VGG5'])))

# Model configration
model_cfg = {
	# (Type, in_channels, out_channels, kernel_size, out_size(c_out*h*w), flops(c_out*h*w*k*k*c_in))
	'VGG5' : [
    ('C', 3, 32, 3, 32*32*32, 32*32*32*3*3*3), # 0
    ('M', 32, 32, 2, 32*16*16, 0),  # 1
	('C', 32, 64, 3, 64*16*16, 64*16*16*3*3*32), #2
    ('M', 64, 64, 2, 64*8*8, 0), # 3
	('C', 64, 64, 3, 64*8*8, 64*8*8*3*3*64), # 4
	('D', 8*8*64, 128, 1, 64, 128*8*8*64), # 5
	('D', 128, 10, 1, 10, 128*10)],# 6
	'VGG9':[#[1,4,7,9,10,11,12,13]
		('C',3,64,3,65535,1769472), 
		('M',64,64,2,16384,0), # 1
		('C',64,64,3,16384,9437184), 
		('C',64,128,3,32768,18874368), 
		('M',128,128,2,8192,0),# 4
		('C',128,128,3,8192,9437184),
		('C',128,256,3,16384,18874368),
		('M',256,256,2,4096,0), #7
		('C',256,256,3,4096,9437184),
		('M',256,256,2,1024,0), #9
		('C',256,256,3,1024,2359296), #10
		('D',1024,4096,1,4096,4194304), # 11
		('D',4096,4096,1,4096,16777216),  # 12
		('D',4096,10,1,10,40960), # 13
	]
}
model_name = 'VGG5'
# model_name = 'VGG9'

if model_name == 'VGG5':
	actionList =  [1,3,4,5,6] # vgg5
	model_size = 2.3 # 1.28
	total_flops = 8488192
	split_layer = [6] #Initial split layers # ? 6
	model_len = 7

	privacy_leakage = [0.552432027525534,0.4516166306393763,0.3399455727858786,0.05740828017257315,0.025361912312856838]
	layer_info = { # action No. : [Flops, smashed_datasize,/ client_side_model_size]
		# 0:[884736,32768],
		0:[884736,8192], # 1
		# 2:[5603328,16384],
		1:[5603328,4096], # 3
		2:[7962624,4096], # 4
		3:[8486912,128], # 5
		4:[8488192,10], # 6
	}

elif model_name == 'VGG9':
	actionList =  [1,4,7,9,10,11,12,13] # vgg9
	model_size = 91.1
	total_flops = 91201536
	split_layer = [13] #Initial split layers # ? 6
	model_len = 14

	privacy_leakage = [0.552432027525534,0.4516166306393763,0.3399455727858786,0.05740828017257315,0.025361912312856838,0.025361912312856838,0.025361912312856838,0.025361912312856838]
	# vgg9[1,4,7,9,10,11,12,13] #8
	layer_info ={
		0:[1769472,16384],
		1:[30081024,8192],
		2:[58392576,4096],
		3:[67829760,1024],
		4:[70189056,1024],
		5:[74383360,4096],
		6:[91160576,4096],
		7:[91201536,10],
	}


# Build the VGG model according to location and split_layer
class VGG(nn.Module):
	def __init__(self, location, vgg_name, split_layer, cfg):
		super(VGG, self).__init__()
		assert split_layer < len(cfg[vgg_name])
		self.split_layer = split_layer
		self.location = location
		self.features, self.denses = self._make_layers(cfg[vgg_name])
		self._initialize_weights()

	def forward(self, x):
		if len(self.features) > 0:
			out = self.features(x)
		else:
			out = x
		if len(self.denses) > 0:
			out = out.view(out.size(0), -1)
			out = self.denses(out)

		return out

	def _make_layers(self, cfg):
		features = []
		denses = []
		if self.location == 'Server':
			cfg = cfg[self.split_layer+1 :]
			
		if self.location == 'Client':
			cfg = cfg[:self.split_layer+1]

		if self.location == 'Unit': # Get the holistic model
			pass

		for x in cfg:
			in_channels, out_channels = x[1], x[2]
			kernel_size = x[3]
			if x[0] == 'M':
				features += [nn.MaxPool2d(kernel_size=kernel_size, stride=2)]
			if x[0] == 'D':
				denses += [nn.Linear(in_channels,out_channels),]
			   				# nn.Tanh()] 

			if x[0] == 'C':
				features += [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=1),
						   nn.BatchNorm2d(out_channels),
						#    nn.ReLU(inplace=True)
						   nn.Tanh()
						#    nn.Sigmoid()
                           ]

		return nn.Sequential(*features), nn.Sequential(*denses)

	def _initialize_weights(self): # 
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
				if m.bias is not None:
					nn.init.constant_(m.bias, 0)
			elif isinstance(m, nn.BatchNorm2d):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)
			elif isinstance(m, nn.Linear):
				nn.init.normal_(m.weight, 0, 0.01)
				nn.init.constant_(m.bias, 0)

	def get_layer_output(self, layer_num, x):
		if layer_num < 0 or layer_num >= len(self.features):
			raise ValueError("Invalid layer number")

		out = x
		for i, layer in enumerate(self.features):
			out = layer(out)
			if i == layer_num:
				return out

		return None

class VGG5Decoder(nn.Module): # 这个cfg参数指的是model_cfg[vgg5]
    def __init__(self,split_layer,network='VGG5'):
        super().__init__()
        cfg = model_cfg[network]
        assert split_layer < len(cfg)
        self.split_layer = split_layer
        self.cfg = cfg[:self.split_layer+1][::-1]
        self.network_name = network

        # print("self.cfg:")
        # print(self.cfg)

        self.features, self.denses = self._make_layers(self.cfg)
        self._initialize_weights()

    def forward(self, x):
        out = x
        if len(self.denses) > 0:
            out = out.view(out.size(0), -1)
            out = self.denses(out)
            if self.network_name == 'VGG5':
                out = out.view([out.size(0),64,8,8]) # vgg5
            else:
                out = out.view([out.size(0),256,2,2]) # vgg9
                 
        if len(self.features) > 0:
            out = self.features(out)
        return out

    def _make_layers(self, cfg):
        features = []
        denses = []
        m = 0
        for x in self.cfg: # 整个网络结构反过来
		# 输入是smashed data
            in_channels, out_channels = x[2], x[1]
            kernel_size = x[3]
            if x[0] == 'M':
                m = 1
            if x[0] == 'D':
                denses += [nn.Linear(in_channels,out_channels),]
						#    nn.Tanh()]
            if x[0] == 'C':
                if m == 1:
                    features += [nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride = 2,padding=1,output_padding=m),
                                nn.BatchNorm2d(out_channels),
                                # nn.ReLU(inplace=True)]
                                nn.Tanh()]
                    m = 0
                else: # m = 0 
                    features += [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=1),
                        nn.BatchNorm2d(out_channels),
                        # nn.ReLU(inplace=True)]
                        nn.Tanh()]

        return nn.Sequential(*features), nn.Sequential(*denses)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


class VGGBranchy(nn.Module): # TODO:
	def __init__(self,vgg_name,split_layer,cfg):
		super().__init__()
		assert split_layer<len(cfg[vgg_name])
		self.split_layer = split_layer
		self.features, self.denses = self._make_layers(cfg[vgg_name])