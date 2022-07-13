import os, time
import numpy as np
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from model import factorized

class  ImageCoder():
    """encode/decode feature using learned entropy model
    """
    def __init__(self, filename, entropy_model):
        self.filename = filename
        self.entropy_model = entropy_model.cpu()

    def encode(self, feats, postfix=''):
        strings, minima, maxima = self.entropy_model.compress(feats.cpu())
        shape = feats.shape
        with open(self.filename+postfix+'_F.bin', 'wb') as fout:
            fout.write(strings)
        with open(self.filename+postfix+'_H.bin', 'wb') as fout:
            fout.write(np.array(shape, dtype=np.int32).tobytes())
            fout.write(np.array(len(minima), dtype=np.int8).tobytes())
            #记录编码元素的属性：最大值，最小值，存储一个float32需要4Bytes
            fout.write(np.array(minima, dtype=np.float32).tobytes())
            fout.write(np.array(maxima, dtype=np.float32).tobytes())
        return 

    def decode(self, postfix=''):
        with open(self.filename+postfix+'_F.bin', 'rb') as fin:
            strings = fin.read()
        with open(self.filename+postfix+'_H.bin', 'rb') as fin:
            shape = np.frombuffer(fin.read(4*4), dtype=np.int32)
            len_minima = np.frombuffer(fin.read(1), dtype=np.int8)[0]
            minima = np.frombuffer(fin.read(4*len_minima), dtype=np.float32)[0]
            maxima = np.frombuffer(fin.read(4*len_minima), dtype=np.float32)[0]
        feats = self.entropy_model.decompress(strings, minima, maxima, shape, channels=shape[1])
        return feats

class Coder():
    def __init__(self, model, filename):
        self.model = model 
        self.filename = filename
        self.coder = ImageCoder(self.filename, model.entropy_bottleneck)

    @torch.no_grad()
    def encode(self, x, postfix=''):
        start_time = time.time()
        #像素正变换获取特征
        y = self.model.encode(x)
        print('Mod Time:\t', round(time.time() - start_time, 3), 's')
        #熵编码特征
        start_time = time.time()
        self.coder.encode(y, postfix=postfix)
        print('Ad Time:\t', round(time.time() - start_time, 3), 's')
        return y

    @torch.no_grad()
    def decode(self, postfix=''):
        # 熵解码特征y [B,C,H,W]
        y = self.coder.decode(postfix=postfix).to(device)
        # decode label
        # 特征反变换得到像素
        out = self.model.decode(y)
        return out