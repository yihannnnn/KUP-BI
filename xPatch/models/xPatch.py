import torch
import torch.nn as nn
import math

from layers.decomp import DECOMP
from layers.network import Network
# from layers.network_mlp import NetworkMLP # For ablation study with MLP-only stream
# from layers.network_cnn import NetworkCNN # For ablation study with CNN-only stream
from layers.revin import RevIN

class GatedFusion(nn.Module):
    """
    Gated fusion with harmonic residual
    - static gate (default): per-channel learnable gate
    - dynamic gate (optional): sequence-level data-dependent gate
    """
    def __init__(self, channels, CalphaC=0.85, use_dynamic_gate=False):
        super().__init__()
        self.CalphaC = CalphaC
        self.use_dynamic_gate = use_dynamic_gate

        if use_dynamic_gate:
            # dynamic: per-channel 2 -> 1 mapping (shared across time)
            self.fc = nn.Linear(2, 1)
        else:
            # static: one gate per channel
            self.gate = nn.Parameter(torch.zeros(1, 1, channels))  # [1,1,C]

    def forward(self, x_main, x_aux):
        """
        x_main, x_aux: [B, T, C]
        """
        if self.use_dynamic_gate:
            # ---- sequence-level data-dependent gate ----
            # time-wise statistics
            m1 = x_main.mean(dim=1, keepdim=True)  # [B, 1, C]
            m2 = x_aux.mean(dim=1, keepdim=True)   # [B, 1, C]

            gate_in = torch.stack([m1, m2], dim=-1)  # [B, 1, C, 2]
            gamma = torch.sigmoid(self.fc(gate_in))  # [B, 1, C, 1]
            gamma = gamma.squeeze(-1)                # [B, 1, C]
        else:
            # ---- static channel gate ----
            gamma = torch.sigmoid(self.gate)         # [1, 1, C] -> broadcast

        fused = gamma * x_main + (1 - gamma) * x_aux
        return self.CalphaC * x_main + (1 - self.CalphaC) * fused

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()

        # Parameters
        seq_len = configs.seq_len   # lookback window L
        pred_len = configs.pred_len # prediction length (96, 192, 336, 720)
        c_in = configs.enc_in       # input channels
        self.CalphaC= configs.CalphaC
        # Patching
        patch_len = configs.patch_len
        stride = configs.stride
        padding_patch = configs.padding_patch

        # Normalization
        self.revin = configs.revin
        self.revin_layer = RevIN(c_in,affine=True,subtract_last=False)
        self.revin_layer_aux  = RevIN(c_in, affine=True, subtract_last=False) 
        # Moving Average
        self.ma_type = configs.ma_type
        alpha = configs.alpha       # smoothing factor for EMA (Exponential Moving Average)
        beta = configs.beta         # smoothing factor for DEMA (Double Exponential Moving Average)
        self.use_dynamic_gate = configs.use_dynamic_gate
        self.fuse_seasonal = GatedFusion(c_in, self.CalphaC, self.use_dynamic_gate)
        self.fuse_trend    = GatedFusion(c_in, self.CalphaC, self.use_dynamic_gate)
        self.fuse = GatedFusion(c_in, self.CalphaC, self.use_dynamic_gate)

        self.decomp = DECOMP(self.ma_type, alpha, beta)
        self.net = Network(seq_len, pred_len, patch_len, stride, padding_patch)
        # self.net_mlp = NetworkMLP(seq_len, pred_len) # For ablation study with MLP-only stream
        # self.net_cnn = NetworkCNN(seq_len, pred_len, patch_len, stride, padding_patch) # For ablation study with CNN-only stream

    def forward(self, x,aux):
        # x: [Batch, Input, Channel]

        # Normalization
        if self.revin:
            x = self.revin_layer(x, 'norm')
            aux = self.revin_layer_aux(aux , 'norm')
        if self.ma_type == 'reg':   # If no decomposition, directly pass the input to the network
            x=self.fuse(x, aux)
            x = self.net(x, x)
            
            # x = self.net_mlp(x) # For ablation study with MLP-only stream
            # x = self.net_cnn(x) # For ablation study with CNN-only stream
        else:
            seasonal_init, trend_init = self.decomp(x)
            seasonal_init_aux, trend_init_aux = self.decomp(aux)

            seasonal_init = self.fuse_seasonal(seasonal_init, seasonal_init_aux)
            trend_init = self.fuse_trend(trend_init, trend_init_aux)

            x = self.net(seasonal_init, trend_init)
        # Denormalization
        if self.revin:
            x = self.revin_layer(x, 'denorm')

        return x