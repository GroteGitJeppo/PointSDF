import torch.nn as nn

from models.so3 import vnnlayers


class vnn_block(nn.Module):
    def __init__(self, hypers_vnn_block):
        super().__init__()
        layers = hypers_vnn_block['layers']
        data_dim = hypers_vnn_block['data_dim']
        share_nonlinearity = hypers_vnn_block['share_nonlinearity']
        negative_slope = hypers_vnn_block['negative_slope']

        sequence = []
        for i in range(len(layers) - 2):
            d_in = layers[i]
            d_out = layers[i + 1]
            if hypers_vnn_block['hid_layer'] == 'linear':
                conv = vnnlayers.VNLinear(d_in, d_out)
            elif hypers_vnn_block['hid_layer'] == 'relu':
                conv = vnnlayers.VNLinearAndLeakyReLU(
                    d_in, d_out, data_dim[i], share_nonlinearity[i], 'none', negative_slope[i])
            elif hypers_vnn_block['hid_layer'] == 'bn_relu':
                conv = vnnlayers.VNLinearLeakyReLU(
                    d_in, d_out, data_dim[i], share_nonlinearity[i], negative_slope[i])
            else:
                raise NotImplementedError
            sequence.append(conv)

        if hypers_vnn_block['last_layer'] == 'linear':
            conv = vnnlayers.VNLinear(layers[-2], layers[-1])
        elif hypers_vnn_block['last_layer'] == 'relu':
            conv = vnnlayers.VNLinearAndLeakyReLU(
                layers[-2], layers[-1], data_dim[-1], share_nonlinearity[-1], 'none', negative_slope[-1])
        elif hypers_vnn_block['last_layer'] == 'bn_relu':
            conv = vnnlayers.VNLinearLeakyReLU(
                layers[-2], layers[-1], data_dim[-1], share_nonlinearity[-1], negative_slope[-1])
        else:
            raise NotImplementedError
        sequence.append(conv)

        if hypers_vnn_block['pool'] == 'max':
            sequence.append(vnnlayers.VNMaxPool(layers[-1]))
        elif hypers_vnn_block['pool'] == 'mean':
            sequence.append(vnnlayers.mean_pool())

        self.model = nn.Sequential(*sequence)

    def forward(self, x):
        return self.model(x)
