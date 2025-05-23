import pickle
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler, MinMaxScaler

class Dataset_Electricity(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='electricity.csv',
                 target='OT', scale=True, timeenc=0, freq='h'):
        # size [seq_len, label_len, pred_len]
        # info
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        self.dim=size[3]
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
        df_raw = pd.read_csv(self.root_path+'/'+self.data_path, index_col='date', parse_dates=True)
        self.scaler = StandardScaler()
        
        df_raw = pd.DataFrame(df_raw)

        num_train = int(len(df_raw) * 0.7)-self.pred_len-self.seq_len+1
        num_test = int(len(df_raw) * 0.2)
        num_valid = int(len(df_raw) * 0.1)
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_valid, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        df_data = df_raw.values

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data)
            data = self.scaler.transform(df_data)
        else:
            data = df_data

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.mask_data = np.ones_like(self.data_x)
        self.reference = torch.load('./data/TCN/ele_idx_list.pt')
        self.reference=torch.clamp(self.reference,min=0, max=17885)
        

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len + self.pred_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        reference=np.zeros((3*self.pred_len, self.dim))
        reference[:self.pred_len, :]=self.data_x[int(self.reference[3*index])+self.seq_len:int(self.reference[3*index])+self.seq_len+self.pred_len]
        reference[self.pred_len:2*self.pred_len]=self.data_x[int(self.reference[3*index+1])+self.seq_len:int(self.reference[3*index+1])+self.seq_len+self.pred_len]
        reference[2*self.pred_len:3*self.pred_len]=self.data_x[int(self.reference[3*index+2])+self.seq_len:int(self.reference[3*index+2])+self.seq_len+self.pred_len]

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = torch.zeros((seq_x.shape[0], 1))
        seq_y_mark = torch.zeros((seq_x.shape[0], 1))
        

        target_mask = self.mask_data[s_begin:s_end].copy()
        target_mask[-self.pred_len:] = 0. #pred mask for test pattern strategy
        s = {
            'observed_data':seq_x,
            'observed_mask': self.mask_data[s_begin:s_end],
            'gt_mask': target_mask,
            'timepoints': np.arange(self.seq_len + self.pred_len) * 1.0, 
            'feature_id': np.arange(370) * 1.0, 
            'reference': reference, 
        }

        return s
    
    def __len__(self):
        length = max(0, len(self.data_x) - self.seq_len - self.pred_len + 1)
        return length

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

def get_dataloader(device, batch_size=8):
    print("Loading electricity dataset...")
    dataset = Dataset_Electricity(root_path='./data/ts2vec',flag='train',size=[168,0,96, 321])
    print(f"Train dataset length: {len(dataset)}, data shape: {dataset.data_x.shape if hasattr(dataset, 'data_x') else 'N/A'}")
    
    if len(dataset) == 0:
        print("Warning: Train dataset is empty! Creating dummy data for testing.")
        dummy_data = np.random.randn(1000, 321)
        dataset.data_x = dummy_data
        dataset.data_y = dummy_data
        dataset.mask_data = np.ones_like(dummy_data)
        print(f"Created dummy data with shape: {dataset.data_x.shape}")
    
    train_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False)
        
    print("Loading validation dataset...")
    valid_dataset = Dataset_Electricity(root_path='./data/ts2vec',flag='val',size=[168,0,96, 321])
    print(f"Validation dataset length: {len(valid_dataset)}")
    
    if len(valid_dataset) == 0:
        print("Warning: Validation dataset is empty! Creating dummy data for testing.")
        dummy_data = np.random.randn(1000, 321)
        valid_dataset.data_x = dummy_data
        valid_dataset.data_y = dummy_data
        valid_dataset.mask_data = np.ones_like(dummy_data)
        
    valid_loader = DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=False)
        
    print("Loading test dataset...")
    test_dataset = Dataset_Electricity(root_path='./data/ts2vec',flag='test',size=[168,0,96,321])
    print(f"Test dataset length: {len(test_dataset)}")
    
    if len(test_dataset) == 0:
        print("Warning: Test dataset is empty! Creating dummy data for testing.")
        dummy_data = np.random.randn(1000, 321)
        test_dataset.data_x = dummy_data
        test_dataset.data_y = dummy_data
        test_dataset.mask_data = np.ones_like(dummy_data)
        
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, valid_loader, test_loader