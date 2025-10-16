#coding: utf-8

import time
import warnings
import argparse
import matplotlib.pyplot as plt
from models.models import *
from collections import OrderedDict
import torch
from torch.utils.data import Dataset, DataLoader
from utils.data_process import *


class EDFDataset(Dataset):
    def __init__(self, data):
        self.data = normalize_within_sample(data)
        self.time_freq_data_3d = self.preprocess_data()

    def preprocess_data(self):
        time_freq_data_3d = []
        for eeg_data in self.data:
            channel_data = []
            for channel in eeg_data:

                f, t, time_freq = signal.spectrogram(channel, fs=256, nperseg=64, noverlap=32, nfft=256, window='hann')
                channel_data.append(time_freq)
            time_freq_data_3d.append(np.stack(channel_data, axis=0))
        time_freq_data_3d = np.expand_dims(np.array(time_freq_data_3d), axis=1)
        return time_freq_data_3d

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        time_freq_data_3d = self.time_freq_data_3d[idx]
        return {'data': torch.tensor(time_freq_data_3d, dtype=torch.float32)}





def test_model(model, test_loader, device):


    model.eval()
    all_preds = []

    with torch.no_grad():
        for batch in test_loader:
            inputs = batch['data'].to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())

    return np.array(all_preds)
def predict(args):
    model_path = args.model
    input_path = args.i
    output_path = args.o

    X_test = process_edf_fixed_params(input_path)
    test_dataset = EDFDataset(X_test)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


    model = SEFRBNet()
    state_dict = torch.load(model_path)
    new_state_dict = OrderedDict()

    for k, v in state_dict.items():
        if k.startswith('module.'):
            name = k[7:]
        else:
            name = k
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict)
    model.to(device)


    raw_predictions = test_model(model, test_loader, device)


    _, alarm_indices = k_of_n_postprocessing(
        raw_predictions, window_size=10, threshold=8
    )
    print(f"Alarms triggered: {len(alarm_indices)}")



  


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parser = argparse.ArgumentParser(description="Test SEFRBNet model and plot results.")
    parser.add_argument('--model', type=str, required=True, help="Path to the model directory.")
    parser.add_argument('--i',type=str, required=True, help="path of the edf.")
    parser.add_argument('--o', type=str, required=True, help="Path to the output directory.")
    args = parser.parse_args()

    predict(args)






