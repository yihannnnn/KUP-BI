import os
import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
import warnings
import torch.nn as nn

warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def _freq2pandas(freq: str) -> str:
    table = {
        'h': 'H',    # 小时
        't': 'T',    # 分钟 (minute)
        '15min': '15T',
        's': 'S',    # 秒
        'd': 'D',    # 天
        'm': 'M',    # 月
    }
    return table.get(freq.lower(), freq)
class Dataset_ETT_hour(Dataset):
    _cache = {}
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h',  
                 top_m: int =3, tau: float =0.05):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.top_m = top_m
        self.tau = tau 
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()
        self.device = device

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)
        #时间戳补充
        aux_len = self.seq_len 
        if aux_len > 0:
            last_date = df_stamp['date'].iloc[-1]
            future_dates = pd.date_range(
                start=last_date + pd.tseries.frequencies.to_offset(_freq2pandas(self.freq)),
                periods=aux_len,
                freq=_freq2pandas(self.freq)
            )

            if self.timeenc == 0:
                f_stamp = np.vstack([
                    future_dates.month,
                    future_dates.day,
                    future_dates.weekday,
                    future_dates.hour
                ]).T.astype('float32')                         # (aux_len, 4)
            else:
                f_stamp = time_features(future_dates, freq=self.freq).T.astype('float32')

            data_stamp = np.vstack([data_stamp, f_stamp])  

        # 开始咯
        self.data_x1 = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp
        self.train_data = data[border1s[0]:border2s[0]]

        TOP_K = 10
        cache_dir = os.path.join(self.root_path, '.cache')
        os.makedirs(cache_dir, exist_ok=True)
        base = os.path.splitext(os.path.basename(self.data_path))[0]
        flag_name = ['train', 'val', 'test'][self.set_type]
        lib_file = os.path.join(
            cache_dir,
            f"lib_{base}_{self.features}_te{self.timeenc}_{self.freq}_{self.seq_len}_{self.pred_len}.npz"
        )
        corr_file = os.path.join(
            cache_dir,
            f"corr_{flag_name}_{base}_{self.features}_te{self.timeenc}_{self.freq}_{self.seq_len}_{self.pred_len}_top{TOP_K}.npz"
        )

        if os.path.exists(lib_file):
            z = np.load(lib_file)
            keys_offset = z['keys_offset']
            self.ratios = z['ratios']
            assert keys_offset.shape == self.ratios.shape, "[lib] keys_offset 与 ratios 形状不一致"

        else:
            # 只基于训练段构建库
            seq_len, pred_len = self.seq_len, self.pred_len
            max_idx = self.train_data.shape[0] - (2 * self.seq_len + self.pred_len) + 1 
            keys = np.zeros((max_idx, self.seq_len, self.train_data.shape[1]))
            values =  np.zeros_like(keys)
            self.ratios = np.zeros_like(keys)
            for idx in range(max_idx):
                keys[idx] = self.train_data[idx: idx + self.seq_len]
                values[idx] = self.train_data[idx + self.seq_len + self.pred_len: idx + 2 * self.seq_len + self.pred_len]
                self.ratios[idx] = (values[idx] - keys[idx]) / (keys[idx] + 1e-4 * np.sign(keys[idx]))
            keys_offset = keys - keys[:, -1:, :] 
            np.savez(lib_file, keys_offset=keys_offset, ratios=self.ratios)
            print(f"[lib] Built & cached: {lib_file}")


        C = self.data_x1.shape[1]
        seq_len = self.seq_len
        max_idx1 = self.data_x1.shape[0] - seq_len + 1

        need_rebuild = True
        if os.path.exists(corr_file):
            z = np.load(corr_file)
            self.corr_vals = z['corr_vals']
            self.corr_idxs = z['corr_idxs']
            if (self.corr_vals.shape == (max_idx1, C, TOP_K) and
                self.corr_idxs.shape == (max_idx1, C, TOP_K)):
                need_rebuild = False

            else:
                print(f"[corr:{flag_name}] Cache shape mismatch, rebuilding…")
                try:
                    os.remove(corr_file)
                except:
                    pass

        if need_rebuild:
            if max_idx1 <= 0:
                raise ValueError(f"[corr:{flag_name}] 当前 split 长度不足以滑窗: N_split={self.data_x1.shape[0]}")
            data_windows = np.array([self.data_x1[i: i + seq_len] for i in range(max_idx1)])  # (max_idx1, seq_len, C)
            dw_offset = data_windows - data_windows[:, -1:, :]

            self.corr_vals = np.zeros((max_idx1, C, TOP_K), dtype=np.float32)
            self.corr_idxs = np.zeros((max_idx1, C, TOP_K), dtype=np.int64)

            for f in range(C):
                x_off = dw_offset[:, :, f]     # (max_idx1, seq_len)
                k_off = keys_offset[:, :, f]   # (max_idx,  seq_len)

                x_mean = x_off.mean(axis=1, keepdims=True)     # (max_idx1,1)
                k_mean = k_off.mean(axis=1, keepdims=True)     # (max_idx,1)
                x_std  = x_off.std(axis=1, keepdims=True)      # (max_idx1,1)
                k_std  = k_off.std(axis=1, keepdims=True).T    # (1, max_idx)

                cov_matrix = (x_off - x_mean) @ (k_off - k_mean).T   # (max_idx1, max_idx)
                denom      = (x_std @ k_std) * seq_len + 1e-4
                corrs      = cov_matrix / denom

                for i in range(max_idx1):
                    row = corrs[i]                      # (max_idx,)
                    eps = 0.0  
                    pos = np.where(row > eps)[0]

                    if pos.size >= TOP_K:
                        pos_sorted = pos[np.argsort(-row[pos])]
                        take = pos_sorted[:TOP_K]
                        self.corr_vals[i, f, :] = row[take]      
                        self.corr_idxs[i, f, :] = take
                    else:
                        take = np.argsort(-row)[:TOP_K]
                        vals = row[take].copy()
                        vals[vals <= eps] = 0.0                
                        self.corr_vals[i, f, :] = vals
                        self.corr_idxs[i, f, :] = take


            np.savez(corr_file, corr_vals=self.corr_vals, corr_idxs=self.corr_idxs)
            print(f"[corr:{flag_name}] Built & cached: {corr_file}")

        index_col = np.arange(1, self.data_x1.shape[0] + 1).reshape(-1, 1)
        self.data_x = np.hstack([index_col, self.data_x1])

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x1 = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        i = int(seq_x1[0, 0])-1 
        corr_full = self.corr_vals[i].T
        idx_full  = self.corr_idxs[i].T
        F = corr_full.shape[1]

        order = np.argsort(-corr_full, axis=0)                 # (TOP_K, F)
        sel   = order[:self.top_m, np.arange(F)]                       # (top_m, F)
        corr_matrix = np.take_along_axis(corr_full, sel, axis=0)       # (top_m, F)
        idx_matrix  = np.take_along_axis(idx_full,  sel, axis=0)       # (top_m, F)

        scaled_corr = corr_matrix / self.tau        
        mask = (corr_matrix > 0)
        scaled_corr[~mask] = -1e9
        scaled_corr = scaled_corr - np.max(scaled_corr, axis=0, keepdims=True)
        weights = torch.softmax(torch.from_numpy(scaled_corr).float(), dim=0)  # (top_m, F)

        coeff_tilde = np.zeros((self.seq_len, F), dtype=np.float32)
        w_np = weights.cpu().numpy()
        for f in range(F):
            if not mask[:, f].any():    
                continue
            ks = idx_matrix[:, f]
            r  = self.ratios[ks, :, f]
            coeff_tilde[:, f] = (w_np[:, [f]] * r).sum(axis=0)
        flat = np.abs(coeff_tilde).flatten()     
        R = max(np.quantile(flat, 0.9), 1e-6)
        coeff_clamped = np.tanh(coeff_tilde / R) * R
        
        hist = seq_x1[:, 1:]  # (seq_len, F)
        aux_input = hist + (coeff_clamped + 1e-4 * np.sign(coeff_clamped)) * hist
        mean_hist = hist.mean(axis=0, keepdims=True)
        std_hist  = hist.std(axis=0, ddof=0, keepdims=True)
        mean_aux  = aux_input.mean(axis=0, keepdims=True)
        std_aux   = aux_input.std(axis=0, ddof=0, keepdims=True)
        aux_input1 = (aux_input - mean_aux) / (std_aux + 1e-4) * (std_hist + 1e-4) + mean_hist

        seq_x_tensor = torch.from_numpy(hist).float()
        aux_tensor   = torch.from_numpy(aux_input1).float()
        seq_y        = torch.from_numpy(seq_y).float()
        seq_x_mark   = torch.from_numpy(seq_x_mark).float()
        seq_y_mark   = torch.from_numpy(seq_y_mark).float()


        ###产生时间戳
        j=int(seq_x1[0, 0])
        aux_mark = self.data_stamp[j + self.seq_len + self.pred_len-1:j + self.seq_len-1 + self.pred_len+ seq_x1.shape[0]]          
        return seq_x_tensor,aux_tensor, seq_y, seq_x_mark, seq_y_mark




    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_ETT_minute(Dataset):
    _cache = {}
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTm1.csv',
                 target='OT', scale=True, timeenc=0, freq='t',  
                 top_m: int =3, tau: float =0.05):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.top_m = top_m
        self.tau = tau 
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()
        self.device = device

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
        border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        #时间戳补充
        aux_len = self.seq_len
        if aux_len > 0:
            last_date = df_stamp['date'].iloc[-1]


            inferred = pd.infer_freq(df_stamp['date'])
            if inferred is not None:
                step = pd.tseries.frequencies.to_offset(inferred)
            else:

                if len(df_stamp['date']) >= 2:
                    step = df_stamp['date'].iloc[-1] - df_stamp['date'].iloc[-2]
                else:
                    step = pd.Timedelta(minutes=15)
                if step <= pd.Timedelta(0):
                    step = pd.Timedelta(minutes=15)


            future_dates = pd.date_range(
                start=last_date + step,
                periods=aux_len,
                freq=step
            )

            if self.timeenc == 0:

                f_stamp = np.vstack([
                    future_dates.month,
                    future_dates.day,
                    future_dates.weekday,
                    future_dates.hour,
                    future_dates.minute // 15
                ]).T.astype('float32')
            else:

                f_stamp = time_features(future_dates, freq=self.freq).T.astype('float32')

            data_stamp = np.vstack([data_stamp, f_stamp])  
        #if True:  # 调试用
            #print("=== data_stamp 最后 5 个 ===")
            #print(df_stamp['date'].tail())

            #print("\n=== f_stamp 对应的 future_dates 前 5 个 ===")
            #print(future_dates[:5])

        # 开始咯
        self.data_x1 = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp
        self.train_data = data[border1s[0]:border2s[0]]

        TOP_K = 10
        cache_dir = os.path.join(self.root_path, '.cache')
        os.makedirs(cache_dir, exist_ok=True)
        base = os.path.splitext(os.path.basename(self.data_path))[0]
        flag_name = ['train', 'val', 'test'][self.set_type]
        lib_file = os.path.join(
            cache_dir,
            f"lib_{base}_{self.features}_te{self.timeenc}_{self.freq}_{self.seq_len}_{self.pred_len}.npz"
        )
        corr_file = os.path.join(
            cache_dir,
            f"corr_{flag_name}_{base}_{self.features}_te{self.timeenc}_{self.freq}_{self.seq_len}_{self.pred_len}_top{TOP_K}.npz"
        )

        if os.path.exists(lib_file):
            z = np.load(lib_file)
            keys_offset = z['keys_offset']
            self.ratios = z['ratios']
            assert keys_offset.shape == self.ratios.shape, "[lib] keys_offset 与 ratios 形状不一致"

        else:
            # 只基于训练段构建库
            seq_len, pred_len = self.seq_len, self.pred_len
            max_idx = self.train_data.shape[0] - (2 * self.seq_len + self.pred_len) + 1 
            keys = np.zeros((max_idx, self.seq_len, self.train_data.shape[1]))
            values =  np.zeros_like(keys)
            self.ratios = np.zeros_like(keys)
            for idx in range(max_idx):
                keys[idx] = self.train_data[idx: idx + self.seq_len]
                values[idx] = self.train_data[idx + self.seq_len + self.pred_len: idx + 2 * self.seq_len + self.pred_len]
                self.ratios[idx] = (values[idx] - keys[idx]) / (keys[idx] + 1e-4 * np.sign(keys[idx]))
            keys_offset = keys - keys[:, -1:, :] 
            np.savez(lib_file, keys_offset=keys_offset, ratios=self.ratios)
            print(f"[lib] Built & cached: {lib_file}")


        C = self.data_x1.shape[1]
        seq_len = self.seq_len
        max_idx1 = self.data_x1.shape[0] - seq_len + 1

        need_rebuild = True
        if os.path.exists(corr_file):
            z = np.load(corr_file)
            self.corr_vals = z['corr_vals']
            self.corr_idxs = z['corr_idxs']
            if (self.corr_vals.shape == (max_idx1, C, TOP_K) and
                self.corr_idxs.shape == (max_idx1, C, TOP_K)):
                need_rebuild = False

            else:
                print(f"[corr:{flag_name}] Cache shape mismatch, rebuilding…")
                try:
                    os.remove(corr_file)
                except:
                    pass

        if need_rebuild:
            if max_idx1 <= 0:
                raise ValueError(f"[corr:{flag_name}] 当前 split 长度不足以滑窗: N_split={self.data_x1.shape[0]}")
            data_windows = np.array([self.data_x1[i: i + seq_len] for i in range(max_idx1)])  # (max_idx1, seq_len, C)
            dw_offset = data_windows - data_windows[:, -1:, :]

            self.corr_vals = np.zeros((max_idx1, C, TOP_K), dtype=np.float32)
            self.corr_idxs = np.zeros((max_idx1, C, TOP_K), dtype=np.int64)

            for f in range(C):
                x_off = dw_offset[:, :, f]     # (max_idx1, seq_len)
                k_off = keys_offset[:, :, f]   # (max_idx,  seq_len)

                x_mean = x_off.mean(axis=1, keepdims=True)     # (max_idx1,1)
                k_mean = k_off.mean(axis=1, keepdims=True)     # (max_idx,1)
                x_std  = x_off.std(axis=1, keepdims=True)      # (max_idx1,1)
                k_std  = k_off.std(axis=1, keepdims=True).T    # (1, max_idx)

                cov_matrix = (x_off - x_mean) @ (k_off - k_mean).T   # (max_idx1, max_idx)
                denom      = (x_std @ k_std) * seq_len + 1e-4
                corrs      = cov_matrix / denom

                for i in range(max_idx1):
                    row = corrs[i]                      # (max_idx,)
                    eps = 0.0  
                    pos = np.where(row > eps)[0]

                    if pos.size >= TOP_K:
                        pos_sorted = pos[np.argsort(-row[pos])]
                        take = pos_sorted[:TOP_K]
                        self.corr_vals[i, f, :] = row[take]      
                        self.corr_idxs[i, f, :] = take
                    else:
                        take = np.argsort(-row)[:TOP_K]
                        vals = row[take].copy()
                        vals[vals <= eps] = 0.0                
                        self.corr_vals[i, f, :] = vals
                        self.corr_idxs[i, f, :] = take


            np.savez(corr_file, corr_vals=self.corr_vals, corr_idxs=self.corr_idxs)
            print(f"[corr:{flag_name}] Built & cached: {corr_file}")

        index_col = np.arange(1, self.data_x1.shape[0] + 1).reshape(-1, 1)
        self.data_x = np.hstack([index_col, self.data_x1])

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x1 = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        i = int(seq_x1[0, 0])-1 
        corr_full = self.corr_vals[i].T
        idx_full  = self.corr_idxs[i].T
        F = corr_full.shape[1]

        order = np.argsort(-corr_full, axis=0)                 # (TOP_K, F)
        sel   = order[:self.top_m, np.arange(F)]                       # (top_m, F)
        corr_matrix = np.take_along_axis(corr_full, sel, axis=0)       # (top_m, F)
        idx_matrix  = np.take_along_axis(idx_full,  sel, axis=0)       # (top_m, F)

        scaled_corr = corr_matrix / self.tau        
        mask = (corr_matrix > 0)
        scaled_corr[~mask] = -1e9
        scaled_corr = scaled_corr - np.max(scaled_corr, axis=0, keepdims=True)
        weights = torch.softmax(torch.from_numpy(scaled_corr).float(), dim=0)  # (top_m, F)

        coeff_tilde = np.zeros((self.seq_len, F), dtype=np.float32)
        w_np = weights.cpu().numpy()
        for f in range(F):
            if not mask[:, f].any():    
                continue
            ks = idx_matrix[:, f]
            r  = self.ratios[ks, :, f]
            coeff_tilde[:, f] = (w_np[:, [f]] * r).sum(axis=0)
        flat = np.abs(coeff_tilde).flatten()     
        R = max(np.quantile(flat, 0.9), 1e-6)
        coeff_clamped = np.tanh(coeff_tilde / R) * R
        
        hist = seq_x1[:, 1:]  # (seq_len, F)
        aux_input = hist + (coeff_clamped + 1e-4 * np.sign(coeff_clamped)) * hist
        mean_hist = hist.mean(axis=0, keepdims=True)
        std_hist  = hist.std(axis=0, ddof=0, keepdims=True)
        mean_aux  = aux_input.mean(axis=0, keepdims=True)
        std_aux   = aux_input.std(axis=0, ddof=0, keepdims=True)
        aux_input1 = (aux_input - mean_aux) / (std_aux + 1e-4) * (std_hist + 1e-4) + mean_hist

        seq_x_tensor = torch.from_numpy(hist).float()
        aux_tensor   = torch.from_numpy(aux_input1).float()
        seq_y        = torch.from_numpy(seq_y).float()
        seq_x_mark   = torch.from_numpy(seq_x_mark).float()
        seq_y_mark   = torch.from_numpy(seq_y_mark).float()


        ###产生时间戳
        j=int(seq_x1[0, 0])
        aux_mark = self.data_stamp[j + self.seq_len + self.pred_len-1:j + self.seq_len-1 + self.pred_len+ seq_x1.shape[0]]              
        return seq_x_tensor,aux_tensor, seq_y, seq_x_mark, seq_y_mark


    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Custom(Dataset):
    _cache = {}
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', 
                 top_m: int =3, tau: float =0.05):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.top_m = top_m
        self.tau = tau
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()
        self.device = device


    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        # print(cols)
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            # print(self.scaler.mean_)
            # exit()
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
        self.train_data = data[border1s[0]:border2s[0]] 
        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)
        #时间戳补充
        aux_len = self.seq_len 
        if aux_len > 0:
            last_date = df_stamp['date'].iloc[-1]
            future_dates = pd.date_range(
                start=last_date + pd.tseries.frequencies.to_offset(_freq2pandas(self.freq)),
                periods=aux_len,
                freq=_freq2pandas(self.freq)
            )

            if self.timeenc == 0:
                f_stamp = np.vstack([
                    future_dates.month,
                    future_dates.day,
                    future_dates.weekday,
                    future_dates.hour
                ]).T.astype('float32')                         # (aux_len, 4)
            else:
                f_stamp = time_features(future_dates, freq=self.freq).T.astype('float32')

            data_stamp = np.vstack([data_stamp, f_stamp])  

        # 开始咯
        self.data_x1 = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp
        self.train_data = data[border1s[0]:border2s[0]]

        TOP_K = 10
        cache_dir = os.path.join(self.root_path, '.cache')
        os.makedirs(cache_dir, exist_ok=True)
        base = os.path.splitext(os.path.basename(self.data_path))[0]
        flag_name = ['train', 'val', 'test'][self.set_type]
        lib_file = os.path.join(
            cache_dir,
            f"lib_{base}_{self.features}_te{self.timeenc}_{self.freq}_{self.seq_len}_{self.pred_len}.npz"
        )
        corr_file = os.path.join(
            cache_dir,
            f"corr_{flag_name}_{base}_{self.features}_te{self.timeenc}_{self.freq}_{self.seq_len}_{self.pred_len}_top{TOP_K}.npz"
        )

        if os.path.exists(lib_file):
            z = np.load(lib_file)
            keys_offset = z['keys_offset']
            self.ratios = z['ratios']
            assert keys_offset.shape == self.ratios.shape, "[lib] keys_offset 与 ratios 形状不一致"

        else:
            # 只基于训练段构建库
            seq_len, pred_len = self.seq_len, self.pred_len
            max_idx = self.train_data.shape[0] - (2 * self.seq_len + self.pred_len) + 1 
            keys = np.zeros((max_idx, self.seq_len, self.train_data.shape[1]))
            values =  np.zeros_like(keys)
            self.ratios = np.zeros_like(keys)
            for idx in range(max_idx):
                keys[idx] = self.train_data[idx: idx + self.seq_len]
                values[idx] = self.train_data[idx + self.seq_len + self.pred_len: idx + 2 * self.seq_len + self.pred_len]
                self.ratios[idx] = (values[idx] - keys[idx]) / (keys[idx] + 1e-4 * np.sign(keys[idx]))
            keys_offset = keys - keys[:, -1:, :] 
            np.savez(lib_file, keys_offset=keys_offset, ratios=self.ratios)
            print(f"[lib] Built & cached: {lib_file}")


        C = self.data_x1.shape[1]
        seq_len = self.seq_len
        max_idx1 = self.data_x1.shape[0] - seq_len + 1

        need_rebuild = True
        if os.path.exists(corr_file):
            z = np.load(corr_file)
            self.corr_vals = z['corr_vals']
            self.corr_idxs = z['corr_idxs']
            if (self.corr_vals.shape == (max_idx1, C, TOP_K) and
                self.corr_idxs.shape == (max_idx1, C, TOP_K)):
                need_rebuild = False

            else:
                print(f"[corr:{flag_name}] Cache shape mismatch, rebuilding…")
                try:
                    os.remove(corr_file)
                except:
                    pass

        if need_rebuild:
            if max_idx1 <= 0:
                raise ValueError(f"[corr:{flag_name}] 当前 split 长度不足以滑窗: N_split={self.data_x1.shape[0]}")
            data_windows = np.array([self.data_x1[i: i + seq_len] for i in range(max_idx1)])  # (max_idx1, seq_len, C)
            dw_offset = data_windows - data_windows[:, -1:, :]

            self.corr_vals = np.zeros((max_idx1, C, TOP_K), dtype=np.float32)
            self.corr_idxs = np.zeros((max_idx1, C, TOP_K), dtype=np.int64)

            for f in range(C):
                x_off = dw_offset[:, :, f]     # (max_idx1, seq_len)
                k_off = keys_offset[:, :, f]   # (max_idx,  seq_len)

                x_mean = x_off.mean(axis=1, keepdims=True)     # (max_idx1,1)
                k_mean = k_off.mean(axis=1, keepdims=True)     # (max_idx,1)
                x_std  = x_off.std(axis=1, keepdims=True)      # (max_idx1,1)
                k_std  = k_off.std(axis=1, keepdims=True).T    # (1, max_idx)

                cov_matrix = (x_off - x_mean) @ (k_off - k_mean).T   # (max_idx1, max_idx)
                denom      = (x_std @ k_std) * seq_len + 1e-4
                corrs      = cov_matrix / denom

                for i in range(max_idx1):
                    row = corrs[i]                      # (max_idx,)
                    eps = 0.0  
                    pos = np.where(row > eps)[0]

                    if pos.size >= TOP_K:
                        pos_sorted = pos[np.argsort(-row[pos])]
                        take = pos_sorted[:TOP_K]
                        self.corr_vals[i, f, :] = row[take]      
                        self.corr_idxs[i, f, :] = take
                    else:
                        take = np.argsort(-row)[:TOP_K]
                        vals = row[take].copy()
                        vals[vals <= eps] = 0.0                
                        self.corr_vals[i, f, :] = vals
                        self.corr_idxs[i, f, :] = take


            np.savez(corr_file, corr_vals=self.corr_vals, corr_idxs=self.corr_idxs)
            print(f"[corr:{flag_name}] Built & cached: {corr_file}")

        index_col = np.arange(1, self.data_x1.shape[0] + 1).reshape(-1, 1)
        self.data_x = np.hstack([index_col, self.data_x1])

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x1 = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        i = int(seq_x1[0, 0])-1 
        corr_full = self.corr_vals[i].T
        idx_full  = self.corr_idxs[i].T
        F = corr_full.shape[1]

        order = np.argsort(-corr_full, axis=0)                 # (TOP_K, F)
        sel   = order[:self.top_m, np.arange(F)]                       # (top_m, F)
        corr_matrix = np.take_along_axis(corr_full, sel, axis=0)       # (top_m, F)
        idx_matrix  = np.take_along_axis(idx_full,  sel, axis=0)       # (top_m, F)

        scaled_corr = corr_matrix / self.tau        
        mask = (corr_matrix > 0)
        scaled_corr[~mask] = -1e9
        scaled_corr = scaled_corr - np.max(scaled_corr, axis=0, keepdims=True)
        weights = torch.softmax(torch.from_numpy(scaled_corr).float(), dim=0)  # (top_m, F)

        coeff_tilde = np.zeros((self.seq_len, F), dtype=np.float32)
        w_np = weights.cpu().numpy()
        for f in range(F):
            if not mask[:, f].any():    
                continue
            ks = idx_matrix[:, f]
            r  = self.ratios[ks, :, f]
            coeff_tilde[:, f] = (w_np[:, [f]] * r).sum(axis=0)
        flat = np.abs(coeff_tilde).flatten()     
        R = max(np.quantile(flat, 0.9), 1e-6)
        coeff_clamped = np.tanh(coeff_tilde / R) * R
        
        hist = seq_x1[:, 1:]  # (seq_len, F)
        aux_input = hist + (coeff_clamped + 1e-4 * np.sign(coeff_clamped)) * hist
        mean_hist = hist.mean(axis=0, keepdims=True)
        std_hist  = hist.std(axis=0, ddof=0, keepdims=True)
        mean_aux  = aux_input.mean(axis=0, keepdims=True)
        std_aux   = aux_input.std(axis=0, ddof=0, keepdims=True)
        aux_input1 = (aux_input - mean_aux) / (std_aux + 1e-4) * (std_hist + 1e-4) + mean_hist

        seq_x_tensor = torch.from_numpy(hist).float()
        aux_tensor   = torch.from_numpy(aux_input1).float()
        seq_y        = torch.from_numpy(seq_y).float()
        seq_x_mark   = torch.from_numpy(seq_x_mark).float()
        seq_y_mark   = torch.from_numpy(seq_y_mark).float()
        ###产生时间戳
        j=int(seq_x1[0, 0])
        aux_mark = self.data_stamp[j + self.seq_len + self.pred_len-1:j + self.seq_len-1 + self.pred_len+ seq_x1.shape[0]]            
        return seq_x_tensor,aux_tensor, seq_y, seq_x_mark, seq_y_mark


    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
    


class Dataset_Pred(Dataset):
    def __init__(self, root_path, flag='pred', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, inverse=False, timeenc=0, freq='15min', cols=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['pred']

        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols = cols
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        if self.cols:
            cols = self.cols.copy()
            cols.remove(self.target)
        else:
            cols = list(df_raw.columns)
            cols.remove(self.target)
            cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        border1 = len(df_raw) - self.seq_len
        border2 = len(df_raw)

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            self.scaler.fit(df_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        tmp_stamp = df_raw[['date']][border1:border2]
        tmp_stamp['date'] = pd.to_datetime(tmp_stamp.date)
        pred_dates = pd.date_range(tmp_stamp.date.values[-1], periods=self.pred_len + 1, freq=self.freq)

        df_stamp = pd.DataFrame(columns=['date'])
        df_stamp.date = list(tmp_stamp.date.values) + list(pred_dates[1:])
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = self.data_x[r_begin:r_begin + self.label_len]
        else:
            seq_y = self.data_y[r_begin:r_begin + self.label_len]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
