import torch
import torch.nn as nn


class moving_avg(nn.Module):
    def __init__(self, kernel_size, stride):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        return x.permute(0, 2, 1)


class series_decomp(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean

class GatedFusion(nn.Module):
    def __init__(self, channels, alpha, use_dynamic=False):

        super().__init__()
        self.alpha = alpha
        self.use_dynamic = use_dynamic

        if use_dynamic:
            self.fc = nn.Linear(2, 1) 
        else:
            self.gate = nn.Parameter(torch.zeros(1, channels, 1))

    def forward(self, x_main, x_aux):
        if self.use_dynamic:
            m1 = x_main.mean(dim=-1, keepdim=True)  # [B, C, 1]
            m2 = x_aux.mean(dim=-1, keepdim=True)   # [B, C, 1]
            gate_in = torch.cat([m1, m2], dim=-1)   # [B, C, 2]
            gamma = torch.sigmoid(self.fc(gate_in)) # [B, C, 1]
        else:
            gamma = torch.sigmoid(self.gate)        # [1, C, 1] -> broadcast

        fused = gamma * x_main + (1 - gamma) * x_aux
        return self.alpha * x_main + (1 - self.alpha) * fused
    
class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.individual = configs.individual
        self.channels = configs.enc_in
        self.alpha = configs.alpha
        self.use_dynamic_gate = configs.use_dynamic_gate
        kernel_size = 25
        self.decompsition = series_decomp(kernel_size)
        
        self.fuse_seasonal = GatedFusion(self.channels, self.alpha, self.use_dynamic_gate)
        self.fuse_trend = GatedFusion(self.channels, self.alpha, self.use_dynamic_gate)


        if self.individual:
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()
            for i in range(self.channels):
                self.Linear_Seasonal.append(nn.Linear(self.seq_len, self.pred_len))
                self.Linear_Trend.append(nn.Linear(self.seq_len, self.pred_len))
        else:
            self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len)
            self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)



    def forward(self, x, aux):
        # Decompose main input
        seasonal_init, trend_init = self.decompsition(x)
        seasonal_init, trend_init = seasonal_init.permute(0, 2, 1), trend_init.permute(0, 2, 1)

        # Decompose auxiliary input
        seasonal_aux, trend_aux = self.decompsition(aux)
        seasonal_aux, trend_aux = seasonal_aux.permute(0, 2, 1), trend_aux.permute(0, 2, 1)

        # Gated fusion with harmonic residual
        seasonal_init = self.fuse_seasonal(seasonal_init, seasonal_aux)
        trend_init = self.fuse_trend(trend_init, trend_aux)

        # Predict
        B, C, _ = seasonal_init.shape
        if self.individual:
            seasonal_output = torch.zeros([B, C, self.pred_len], dtype=seasonal_init.dtype, device=seasonal_init.device)
            trend_output = torch.zeros([B, C, self.pred_len], dtype=trend_init.dtype, device=trend_init.device)
            for i in range(self.channels):
                seasonal_output[:, i, :] = self.Linear_Seasonal[i](seasonal_init[:, i, :])
                trend_output[:, i, :] = self.Linear_Trend[i](trend_init[:, i, :])
        else:
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            trend_output = self.Linear_Trend(trend_init)

        return (seasonal_output + trend_output).permute(0, 2, 1)  # [B, L, C]
