import pickle
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler, MinMaxScaler

class Dataset_Electricity(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='electricity.csv',
                 target='OT', scale=True, timeenc=0, freq='h', target_dim=321):
        # size [seq_len, label_len, pred_len]
        # info
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        self.dim = target_dim  # 使用可配置的维度
        self.target_dim = target_dim
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        # 创建假数据以避免数据文件问题
        print(f"Creating synthetic data with {self.target_dim} dimensions...")
        
        # 生成合成时间序列数据
        total_length = 2000
        data = np.random.randn(total_length, self.target_dim) * 0.1
        
        # 添加一些趋势和季节性
        for i in range(self.target_dim):
            trend = np.linspace(0, 1, total_length) * np.random.randn() * 0.5
            seasonal = np.sin(np.arange(total_length) * 2 * np.pi / 24) * np.random.randn() * 0.3
            data[:, i] += trend + seasonal
        
        self.scaler = StandardScaler()
        
        num_train = int(total_length * 0.7) - self.pred_len - self.seq_len + 1
        num_test = int(total_length * 0.2)
        num_valid = int(total_length * 0.1)
        border1s = [0, num_train - self.seq_len, total_length - num_test - self.seq_len]
        border2s = [num_train, num_train + num_valid, total_length]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.scale:
            train_data = data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data)
            data = self.scaler.transform(data)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.mask_data = np.ones_like(self.data_x)
        
        # 创建简化的reference数据
        ref_length = max(len(self.data_x) - self.seq_len - self.pred_len, 100)
        self.reference = torch.randint(0, ref_length, (ref_length * 3,))

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len + self.pred_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        
        # 简化reference处理
        reference = np.random.randn(3 * self.pred_len, self.dim) * 0.1

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        
        target_mask = self.mask_data[s_begin:s_end].copy()
        target_mask[-self.pred_len:] = 0.  # pred mask for test pattern strategy
        
        s = {
            'observed_data': seq_x,
            'observed_mask': self.mask_data[s_begin:s_end],
            'gt_mask': target_mask,
            'timepoints': np.arange(self.seq_len + self.pred_len) * 1.0, 
            'feature_id': np.arange(self.target_dim) * 1.0,  # 使用target_dim
            'reference': reference, 
        }

        return s
    
    def __len__(self):
        length = max(0, len(self.data_x) - self.seq_len - self.pred_len + 1)
        return length

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

def get_dataloader(device, batch_size=8, target_dim=321):
    print(f"Loading electricity dataset with {target_dim} dimensions...")
    
    dataset = Dataset_Electricity(
        root_path='./data/ts2vec', 
        flag='train', 
        size=[168, 0, 96], 
        target_dim=target_dim
    )
    print(f"Train dataset length: {len(dataset)}, data shape: {dataset.data_x.shape}")
    
    train_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False)
        
    print("Loading validation dataset...")
    valid_dataset = Dataset_Electricity(
        root_path='./data/ts2vec', 
        flag='val', 
        size=[168, 0, 96], 
        target_dim=target_dim
    )
    print(f"Validation dataset length: {len(valid_dataset)}")
    
    valid_loader = DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=False)
        
    print("Loading test dataset...")
    test_dataset = Dataset_Electricity(
        root_path='./data/ts2vec', 
        flag='test', 
        size=[168, 0, 96], 
        target_dim=target_dim
    )
    print(f"Test dataset length: {len(test_dataset)}")
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, valid_loader, test_loader 