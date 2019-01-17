#   Base lightnet network module structure
#   Copyright EAVISE

import logging as log
import torch
import torch.nn as nn
import time
# 返回当前的时间戳

__all__ = ['Lightnet']


class Lightnet(nn.Module):
    """ This class provides an abstraction layer on top of :class:`pytorch:torch.nn.Module` and is used as a base for every network implemented in this framework.
    There are 2 basic ways of using this class:
    - Override the ``forward()`` function.
      This makes :class:`lightnet.network.Lightnet` networks behave just like PyTorch modules.
    - Define ``self.loss`` and ``self.postprocess`` as functions and override the ``_forward()`` function.
      This class will then automatically call the loss and postprocess functions on the output of ``_forward()``,
      depending whether the network is training or evaluating.
    Attributes:
        self.seen (int): The number of images the network has processed to train *(used by engine)*
    Note:
        If you define **self.layers** as a :class:`pytorch:torch.nn.Sequential` or :class:`pytorch:torch.nn.ModuleList`,
        the default ``_forward()`` function can use these layers automatically to run the network.
    Warning:
        If you use your own ``forward()`` function, you need to update the **self.seen** parameter
        whenever the network is training.
    """
    def __init__(self):  # __代表特殊函数
        super().__init__()
        # super( test, self).__init__()
        # 首先找到test的父类（比如是类A），
        # 然后把类test的对象self转换为类A的对象，然后“被转换”的类A对象调用自己的__init__函数
        # Parameters
        self.layers = None
        self.loss = None
        self.postprocess = None
        self.seen = 0  # 这个是干嘛？

    def _forward(self, x):  # 感觉只是用来判断而已 _代表私有函数
        log.debug('Running default forward function')  # 打印到控制台
        if isinstance(self.layers, nn.Sequential):
            return self.layers(x)
        # isinstance 判断一个对象是否和后面的类型相同
        elif isinstance(self.layers, nn.ModuleList):
            log.warning('No _forward function defined, looping sequrentially over modulelist')
            for _, module in enumerate(self.layers):
                x = module(x)
            return x
            # 把module存入x
        else:
            raise NotImplementedError(f'No _forwad function defined and no default behaviour for this type of layers [{type(self.layers)}]')

    def forward(self, x, target=None):
    # This default forward function will compute the output of the network as ``self._forward(x)``.
    # Then, depending on whether you are training or evaluating,
    # it will pass that output to ``self.loss()`` or ``self.posprocess()``.
    # This function also increments the **self.seen** variable.
        if self.training:
            self.seen += x.size(0)  # seen在上面定义了
            t1 = time.time()
            output = self._forward(x)
            t2 = time.time()

            assert len(output) == len(self.loss)  # 判断output 长度和self.loss长度

            loss = 0
            for idx in range(len(output)):  # 遍历每一层 output = self.forward
                assert callable(self.loss[idx])  # callable内置函数 检测对象是否可以被调用
                t1 = time.time()
                loss += self.loss[idx](output[idx], target)
                t2 = time.time()
            return loss
        else:
            output = self._forward()
            if self.postprocess is None:
                return  # speed
            loss = None
            dets  = []
            tdets = []










