# base network structure for Darknet networks
# copyright EAVISE

import os
import collections
import logging as log
# 内置标准模块 主要用于输出运行日志
import numpy as np
import torch
import torch.nn as nn
from _lightnet import Lightnet

# from .network import layer as vn_layer
import sys
sys.path.append('..')
from network import layer as vn_layer

__all__ = ['Darknet']


class Darknet(Lightnet):

    def __init__(self):
        super().__init__()
        self.header = [0, 2, 0]

    # 加载模型，分为torch和darknet
    def load_weights(self, weights_file, clear):
        if os.path.splitext(weights_file)[1] == '.pt':
            log.debug('Loading weights from pytorch file')
            super().load_weights(weights_file, clear)
        else:  # darknet weight
            log.debug('Loading weight from darknet file')
            self._load_darknet_weights(weights_file, clear)

    # 保存模型，同样分为两种
    def save_weights(self, weights_file, seen=None):
        if os.path.splitext(weights_file)[1] == '.pt':
            log.debug('save weight to pytorch file')
            super().save_weights(weights_file,seen)  #在lightnet类里面
        else:
            log.debug('save weight to darknet file')
            self._save                           # 自定义的函数在下面


    def _load_darknet_weights(self, weights_file, clear=False):
        weights = WeightLoader(weights_file)
        self.header = weights.header
        self.seen = weights.seen
        if clear:
            self.seen = 0
        if hasattr(self.loss, 'seen'):
            self.loss.seen = self.seen

        for module in self.modules_recurse():
            try:
                weights.load_layer(module)
                log.debug(f'Layer loaded: {module}')
                if weights.start>=weights.size:
                    log.debug(f'Finished loading weights [{weights.start}/{weights.size} weights]')
                    break
            except NotImplementedError:
                log.debug(f'Layer skipped: {module.__class__.__name__}')

    def _save_darknet_weight(self, weight_file, seen=None):
        if seen is None:
            seen = self.seen
        weights = WeightSaver(self.header, seen)
        # WeightServer类。在下文中定义
        for module in self.modules_recurse():
            try:
                weights.save_layer(module)
                log.debug(f'layer saved:{module}')
            except NotImplementedError:
                log.debug(f'layer skipped: {module.__class__.__name__}')
        weights.write_file(weight_file)



class WeightLoader:
    def __init__(self, filename):
        with open(filename, 'rb') as fp:
            self.header = np.fromfile(fp, count=3, dtype=np.int32).tolist()
            ver_num = self.header[0]*100+self.header[1]*10+self.header[2]
            log.debug(f'Loading weight file:version{self.header[0]}.{self.header[1]}.{self.header[2]}')

            if ver_num <= 19:
                log.warning('Weight file uses sizeof to compute variable size, which might'
                            'lead to undefined behaviour.'
                            '(choosing int=int32, float=float32)')
                self.seen = int(np.fromfile(fp, count=1, dtype=np.int32)[0])
            elif ver_num <=29:
                log.warning('Weight file uses sizeof to compute variable size, which might'
                            'lead to undefined behaviour.'
                            '(choosing int=int32, float=float32, size_t=int64)')
                self.seen = int(np.fromfile(fp, count=1, dtype=np.int64)[0])
            else:
                log.error('New weight file syntax! loading of weights might not work properly.'
                          'Please submit an issue with the weight file version number.'
                          '[Run with DEBUG logging level]')
                self.seen = int(np.fromfile(fp, count=1, dtype=np.int64)[0])

            self.buf = np.fromfile(fp,dtype=np.float32)
        self.start = 0
        self.size = self.buf.szie

    def load_layer(self, layer):
        # 从每一层中读取权重
        if type(layer) == nn.Conv2d:
            self._load_conv(layer)  #载入卷积函数，在下文定义
        elif type(layer) == vn_layer.Con2dBatchLeaky:  # 载入Conv2dBatchLeaky 该函数在network-layer-_darknet.py内
            self._load_convbatch(layer)              #load conv batch layer 函数在下面
        elif type(layer) == nn.Linear:
            self._load_fc(layer)                    # 载入全连接层，定义在下面
        else:
            raise NotImplementedError(f'The layer you are trying to load is not supported [{type(layer)}]')

    def _load_conv(self, model):
        num_b = model.bias.numel()  # 该函数没有补全
        model.bias.data.copy_(torch.from_numpy(self.buf[self.start:self.start+num_b])
                              .view_as(model.bias.data))
        # 返回被视作与给定的tensor相同大小的原tensor。等效于： self.view(tensor.size())
        self.start += num_b
        num_w = model.weight.numel()
        model.weight.data.copy_(torch.from_numpy(self.buf[self.start:self.start+num_w])
                                .view_as(model.weight.data))
        self.start +=num_w

    def _load_convbatch(self, model):
        num_b = model.layers[1].bias.numel()
        model.layers[1].bias.data.copy_(torch.from_numpy(self.buf[self.start:self.start+num_b])
                                         .view_as(model.layers[1].bias.data))
        self.start += num_b
        model.layers[1].weight.data.copy_(torch.from_numpy(self.buf[self.start:self.start+num_b])
                                          .view_as(model.layers[1].weight.data))
        self.start += num_b
        model.layers[1].running_mean.copy_(torch.from_numpy(self.buf[self.start:self.start+num_b])
                                           .view_as(model.layers[1].running_mean))
        self.start += num_b
        model.layers[1].running_var.copy_(torch.from_numpy(self.buf[self.start:self.start+num_b])
                                          .view_as(model.layers[1].running_var))
        self.start += num_b

        num_w = model.layers[0].weight.numel
        model.layers[0].weight.data.copy_(torch.from_numpy(self.buf[self.start:self.start+num_w])
                                          .view_ax(model.layers[0].weight.data))
        self.start += num_w

    def _load_fc(self, model):
        num_b = model.bias.numel()
        model.bias.data.copy_(torch.from_numpy(self.buf[self.start:self.start+num_b])
                              .view_as(model.bias.data))
        self.start += num_b

        num_w = model.weight.numel()
        model.weight.data.copy_(torch.from_numpy(self.buf[self.start:self.start+num_w])
                                .view_as(model.weight.data))
        self.start += num_w


class WeightSaver:
    def __init__(self, header, seen):
        self.weights = []
        self.header = np.array(header, dtype=np.int32)
        ver_num = self.header[0]*100 + self.header[1]*10 +self.header[2]
        if ver_num<=19:
            self.seen = np.int32(seen)
        elif ver_num<=29:
            self.seen = np.int64(seen)
        else:
            log.error('New weight file syntax! Saving weight might not work properly')
            self.seen = np.int64(seen)

    def write_file(self,filename):
        log.debug(f'writing weight file : version {self.header[0]}.{self.header[1]},{self.header[2]}')
        with open(filename, 'wb') as fp:
            self.header.tofile(fp)
            self.seen.tofile(fp)
            for np_arr in self.weights:
                np_arr.tofile(fp)
        log.info(f'weight file saved as {filename}')

    def save_layer(self, layer):
        # save weight for layer
        if type(layer) == nn.Conv2d:
            self.__save_conv(layer)
        elif type(layer) == vn_layer.Conv2dBatchLeaky:
            self._save_convdatch(layer)
        elif type(layer) == nn.Linear:
            self._save_fc(layer)
        else:
            raise NotImplementedError(f'the layer you are trying to save is not supported [{type(layer)}]')

    def _save_conv(self,model):
        self.weights.append(model.bias.cpu().data.numpy())
        self.weights.append(model.weight.cpu().data.numpy())

    def _save_convbatch(self,model):
        self.weights.append(model.layers[1].bias.cpu().data.numpy())
        self.weights.append(model.layers[1].weight.cpu().data.numpy())
        self.weights.append(model.layers[1].running_mean.cpu().numpy())
        self.weights.append(model.layers[1].running_var.cpu().numpy())
        self.weights.append(model.layers[0].weight.cpu().data.numpy())

    def _save_fc(self,model):
        self.weights.append(model.bias.cpu().data.numpy())
        self.weights.append(model.weight.cpu().data.numpy)
