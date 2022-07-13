#熵模型
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torchac
import numpy as np

#覆写自动求导的类实现反传梯度为1
class RoundNoGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.round()

    @staticmethod
    def backward(ctx, g):
        return g

#覆写自动求导类实现对x的下界约束
class Low_bound(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        x = torch.clamp(x, min=1e-9)
        return x

    @staticmethod
    def backward(ctx, g):
        x, = ctx.saved_tensors
        grad1 = g.clone()
        try:
            grad1[x<1e-9] = 0
        except RuntimeError:
            print("ERROR! grad1[x<1e-9] = 0")
            grad1 = g.clone()
        pass_through_if = np.logical_or(x.cpu().detach().numpy() >= 1e-9, g.cpu().detach().numpy()<0.0)
        t = torch.Tensor(pass_through_if+0.0).to(grad1.device)

        return grad1*t

class EntropyBottleneck(nn.Module):
    """The layer implements a flexible probability density model to estimate
    entropy of its input tensor, which is described in this paper:
    >"Variational image compression with a scale hyperprior"
    > J. Balle, D. Minnen, S. Singh, S. J. Hwang, N. Johnston
    > https://arxiv.org/abs/1802.01436"""
    
    def __init__(self, channels, init_scale=8, filters=(3,3,3,3)):
        """create parameters.
        """
        super(EntropyBottleneck, self).__init__()
        self._likelihood_bound = 1e-9
        self._init_scale = float(init_scale)
        self._filters = tuple(int(f) for f in filters)
        self._channels = channels
        self.ASSERT = False
        # build.
        filters = (1,) + self._filters + (1,)
        scale = self._init_scale ** (1 / (len(self._filters) + 1))
        # Create variables.
        self._matrices = nn.ParameterList([])
        self._biases = nn.ParameterList([])
        self._factors = nn.ParameterList([])

        for i in range(len(self._filters) + 1):
            #
            self.matrix = Parameter(torch.FloatTensor(channels, filters[i + 1], filters[i]))
            init_matrix = np.log(np.expm1(1.0 / scale / filters[i + 1]))
            self.matrix.data.fill_(init_matrix)
            self._matrices.append(self.matrix)
            #
            self.bias = Parameter(torch.FloatTensor(channels, filters[i + 1], 1))
            init_bias = torch.FloatTensor(np.random.uniform(-0.5, 0.5, self.bias.size()))
            self.bias.data.copy_(init_bias)# copy or fill?
            self._biases.append(self.bias)
            #       
            self.factor = Parameter(torch.FloatTensor(channels, filters[i + 1], 1))
            self.factor.data.fill_(0.0)
            self._factors.append(self.factor)

    def _logits_cumulative(self, inputs):
        """Evaluate logits of the cumulative densities.
        
        Arguments:
        inputs: The values at which to evaluate the cumulative densities,
            expected to have shape `(channels, 1, batch*H*W)`.

        Returns:
        A tensor of the same shape as inputs, containing the logits of the
        cumulatice densities evaluated at the the given inputs.
        """
        logits = inputs
        for i in range(len(self._filters) + 1):
            matrix = torch.nn.functional.softplus(self._matrices[i])
            logits = torch.matmul(matrix, logits)
            logits += self._biases[i]
            factor = torch.tanh(self._factors[i])
            logits += factor * torch.tanh(logits)
        return logits

    def _quantize(self, inputs, mode):
        """Add noise or quantize."""
        if mode == "noise":
            noise = np.random.uniform(-0.5, 0.5, inputs.size())
            noise = torch.Tensor(noise).to(inputs.device)
            return inputs + noise
        if mode == "symbols":
            return RoundNoGradient.apply(inputs)

    def _likelihood(self, inputs):
        #Input:[channels, 1, points] Output:[channels, 1, points]
        lower = self._logits_cumulative(inputs - 0.5)
        upper = self._logits_cumulative(inputs + 0.5)
        sign = -torch.sign(torch.add(lower, upper)).detach()
        likelihood = torch.abs(torch.sigmoid(sign * upper) - torch.sigmoid(sign * lower))
        return likelihood

    def forward(self, inputs, quantize_mode="noise"):
        #Input:[B,C,H,W] --> [C, 1, B*H*W] -->Output:[B,C,H,W]
        #维度转换
        perm = np.arange(len(inputs.shape))
        perm[0], perm[1] = perm[1], perm[0]
        inv_perm = np.arange(len(inputs.shape))[np.argsort(perm)]
        inputs = inputs.permute(*perm).contiguous()
        shape = inputs.size()
        inputs = inputs.reshape(inputs.size(0), 1, -1)
        #量化
        if quantize_mode is None: 
            outputs = inputs
        else: 
            outputs = self._quantize(inputs, mode=quantize_mode)
        likelihood = self._likelihood(outputs)
        likelihood = Low_bound.apply(likelihood)
        #转化为输入维度
        outputs = outputs.reshape(shape)
        outputs = outputs.permute(*inv_perm).contiguous()

        likelihood = likelihood.reshape(shape)
        likelihood = likelihood.permute(*inv_perm).contiguous()
        return outputs, likelihood

    #将概率转换为累积概率
    def _pmf_to_cdf(self, pmf):
        cdf = pmf.cumsum(dim=-1)
        #最后一维变成1
        spatial_dimensions = pmf.shape[:-1]+(1,)
        zeros = torch.zeros(spatial_dimensions, dtype=pmf.dtype, device=pmf.device)
        #cdf第一个元素设置为0
        cdf_with_0 = torch.cat([zeros, cdf], dim=-1)
        #cdf最大值即最后一个值设置为1
        cdf_max_1 = cdf_with_0.clamp(max=1.)
        return cdf_max_1

    @torch.no_grad()
    def compress(self, inputs):
        #Input:[B,C,H,W]
        #量化
        values = self._quantize(inputs, mode="symbols")
        batch_size, channels, H, W = values.shape[:]
        #获得编码范围内所有字符的概率值
        minima = values.min().detach().float()
        maxima = values.max().detach().float()
        symbols = torch.arange(minima, maxima+1)
        # 对编码元素做一个偏移，可选
        values_norm = (values-minima).to(torch.int16)
        minima, maxima = torch.tensor([minima]), torch.tensor([maxima])
        
        #[channels, 1, points],points为编码值范围
        symbols = symbols.reshape(1, 1, -1).repeat(channels, 1, 1)

        pmf = self._likelihood(symbols)
        pmf = torch.clamp(pmf, min=self._likelihood_bound)

        cdf = self._pmf_to_cdf(pmf)
        cdf_expand = cdf.unsqueeze(0).unsqueeze(2)
        out_cdf = cdf_expand.repeat(batch_size, 1, H, W, 1)

        #torchac传入两个主要参数，概率表和待编码元素，
        # 概率表out_cdf是按顺序排列的概率，格式[B,C,H,W,point]，待编码元素格式[B,C,H,W]
        strings = torchac.encode_float_cdf(out_cdf, values_norm)
       
        return strings, minima.cpu().numpy(), maxima.cpu().numpy()

    @torch.no_grad()
    def decompress(self, strings, minima, maxima, shape, channels):
        batch_size, channels, H, W = shape[:]
        symbols = torch.arange(minima, maxima+1)
        symbols = symbols.reshape(1, 1, -1).repeat(channels, 1, 1)

        pmf = self._likelihood(symbols)
        pmf = torch.clamp(pmf, min=self._likelihood_bound)

        cdf = self._pmf_to_cdf(pmf)
        out_cdf = cdf.unsqueeze(0).unsqueeze(2).repeat(batch_size, 1, H, W, 1)
        values = torchac.decode_float_cdf(out_cdf, strings)
        values = values.float()
        values += minima

        return values




