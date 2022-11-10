import os

from torch.utils.data import Dataset
import torchaudio
import torch


# This is a custom dataset class.
class CustomDataset(Dataset):
    def __init__(self, path_dataset, is_train=True,
                 transform=False, max_length=32000, sr=4000):
        self.samples = []
        self.labels_map = {}
        self.is_train = is_train
        self.transform = transform
        self.max_length = max_length
        self.sr = sr
        self.read(path_dataset)

    def read(self, path_dataset):
        for idx_class, class_name in enumerate(os.listdir(path_dataset)):
            path_class = os.path.join(path_dataset, class_name)
            self.labels_map[class_name] = idx_class
            if self.is_train:
                path_class = os.path.join(path_class, "train")
            else:
                path_class = os.path.join(path_class, "test")

            for idx, file_name in enumerate(os.listdir(path_class)):
                path_file = os.path.join(path_class, file_name)
                waveform, sr = torchaudio.load(path_file)
                metadata = torchaudio.info(path_file)
                if self.transform:
                    transform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sr)
                    waveform = transform(waveform)
                    waveform = self.padding(waveform, self.max_length)
                    sr = self.sr

                self.samples.append((waveform, idx_class, sr))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index][0], self.samples[index][1], self.samples[index][2]

    def padding(self, waveform, max_len):
        # Pad the waveform
        length_waveform = waveform.shape[1]
        if length_waveform < max_len:
            waveform = torch.cat((waveform, torch.zeros((1, max_len - length_waveform))), dim=1)
        return waveform

    def getLabelsMap(self):
        return self.labels_map

    def getLabelCount(self):
        return len(self.labels_map)
