import torch
import torch.nn as nn

class conv2d(nn.Module):
    def __init__(self, in_channel, out_channel=None, stride=False):
        super(conv2d, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, in_channel * 2, kernel_size=3, padding=1, bias=False)
        weight = self._init_weight(in_channel, in_channel) # in_channel * 2, in_channel
        conv1_weight = weight.reshape(in_channel * 2, in_channel, 1, 1)
        conv1_weight = nn.ZeroPad2d(1)(conv1_weight)
        self.conv1.weight = nn.Parameter(conv1_weight)
        self.bn1 = nn.BatchNorm2d(in_channel)


        self.conv2 = nn.Conv2d(in_channel, in_channel * 4, kernel_size=1, bias=False)
        weight = self._init_weight(in_channel, in_channel * 2) # in_channel * 4, in_channel
        conv2_weight = weight.reshape(in_channel * 4, in_channel, 1, 1)
        self.conv2.weight = nn.Parameter(conv2_weight.contiguous())

        if out_channel is None:
            out_channel = in_channel
        weight = torch.randn(out_channel, in_channel * 4)
        weight[:in_channel] = conv2_weight[:,:,0,0].transpose(0, 1) / 2 ** .5
        for k in range(in_channel, out_channel):
            r = weight[k] - weight[:k].T @ (weight[:k] @ weight[k])
            weight[k] = r / r.norm()
        self.conv3 = nn.Parameter(weight.view(out_channel, -1, 1, 1).contiguous())
        in_channel = out_channel

        self.bn2 = nn.BatchNorm2d(in_channel)

        if stride:
            self.pool = nn.MaxPool2d(3, 2, 1)

    def forward(self, x):
        # 3x3
        x = self.conv1(x)
        if hasattr(self, 'pool'):
            x = self.pool(x)
        x = x.relu_()
        weight = self.conv1.weight[:,:,1:2,1:2].transpose(0,1)
        x = nn.functional.conv2d(x, weight=weight)
        x = self.bn1(x)

        # 1x1
        x = self.conv2(x)
        x = x.relu_()
        x = nn.functional.conv2d(x, weight=self.conv3)
        x = self.bn2(x)
        return x

    def _init_weight(self, in_channel, mid_channel=None):
        assert mid_channel >= in_channel and mid_channel % in_channel == 0
        num_repeats = mid_channel // in_channel
        W = [torch.randn(in_channel, in_channel) for _ in range(num_repeats)]
        [nn.init.orthogonal_(w) for w in W]
        W = torch.cat(W, dim=1) / num_repeats ** .5
        weight = torch.cat([W, -W], dim=1).transpose(0, 1)
        return weight
