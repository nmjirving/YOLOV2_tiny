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
        # 定义属性
        self.layers = None
        self.loss = None
        self.postprocess = None  # 后处理
        self.seen = 0  # 这个是干嘛？The number of images the network has processed to train

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
        if self.training:  # 如果是训练模式
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
        else:  # 如果不是训练模式
            output = self._forward()
            if self.postprocess is None:
                return  # speed
            loss = None
            dets  = []
            tdets = []
            for idx in range(len(output)):
                assert callable(self.postprocess[idx])
                tdets.append(self.postprocess[idx](output[idx]))

            batch = len(tdets[0])
            for b in range(batch):
                single_dets = []
                for op in range(len(output)):
                    single_dets.extend(tdets[op][b])  # extend() 函数用于在列表末尾一次性追加另一个序列中的多个值
                dets.append(single_dets)

            if loss is not None:
                return dets, loss
            else:
                return dets, 0.0

    def modules_recurse(self, mod=None):  # 迭代
        if mod is None:
            mod = self
        for module in mod.children():
            if isinstance(module, (nn.ModuleList, nn.Sequential)):
                yield from self.modules_recurse(module)
            else:
                yield module

    def init_weights(self, mode='fan_in', slope=0):
        info_list = []
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                info_list.append(str(m))
                nn.init.kaiming_normal_(m.weight, a=slope, mode=mode)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                info_list.append(str(m))
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                info_list.append(str(m))
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
        log.info('Init weight\n\n%s\n' % '\n'.join(info_list))

    def load_weights(self, weights_file, clear=False):
        old_state = self.state_dict()
        state = torch.load(weights_file, lambda storage, loc: storage)
        self.seen = 0 if clear else state['seen']
        self.load_state_dict(state['weights'])
        if hasattr(self.loss, 'seen'):  # 判断对象中是否存在seen属性
            self.loss.seen = self.seen
        log.info(f'Loaded weights from {weights_file}')

    def save_weights(self, weights_file, seen=None):
        if seen is None:
            seen = self.seen
        state = {'seen': seen,
                 'weight': self.state_dict()}
        torch.save(state, weights_file)
        log.info(f'Saved weight as {weights_file}')


