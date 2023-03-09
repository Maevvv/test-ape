import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

#在网络中添加高斯噪声，得到一个新的Q函数进行选择动作，噪声是固定的


class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.4):
        super(NoisyLinear, self).__init__()
        
        # 输入维度
        self.in_features  = in_features
        # 输出维度
        self.out_features = out_features
        # 初始的标准差
        self.std_init     = std_init
        # nn.Parameter()：向模型中注册一个参与梯度计算、参与更新的变量
        self.weight_mu    = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(out_features, in_features))
        # register_buffer()：向模型中注册一个不参与梯度计算、不参与更新的变量
        self.register_buffer('weight_epsilon', torch.FloatTensor(out_features, in_features))
        
        self.bias_mu    = nn.Parameter(torch.FloatTensor(out_features))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(out_features))
        self.register_buffer('bias_epsilon', torch.FloatTensor(out_features))
        
        self.reset_parameters()
        self.reset_noise()
    
    def forward(self, x):

        weight_epsilon = self.weight_epsilon
        bias_epsilon   = self.bias_epsilon
            
        if self.training: 
            weight = self.weight_mu + self.weight_sigma.mul(Variable(weight_epsilon))
            bias   = self.bias_mu   + self.bias_sigma.mul(Variable(bias_epsilon))
        else:
            weight = self.weight_mu
            bias   = self.bias_mu
        
        return F.linear(x, weight, bias)
    
    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.weight_mu.size(1))
        # uniform_()：从均匀分布中抽样数值进行填充
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        # fill_()：用某个数值进行填充
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.weight_sigma.size(1)))
        
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.bias_sigma.size(0)))
    
    def reset_noise(self):
        epsilon_in  = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        # copy_()：将传入tensor对象的参数复制给调用的tensor对象
        # A.ger(B)：将A的元素依次乘以B的元素，进行扩维
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(self._scale_noise(self.out_features))
    
    def _scale_noise(self, size):
        # randn()：产生指定大小的，正态分布的采样点
        x = torch.randn(size)
        # sign()：一个符号函数，>0返回1，=0返回0，<0返回-1
        # mul()：两个同等大小的矩阵进行点乘
        x = x.sign().mul(x.abs().sqrt())
        return x