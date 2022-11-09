import os

from torch.utils.data import Dataset
import torchaudio


# This is a custom dataset class.
class CustomDataset(Dataset):
    def __init__(self, path_dataset, is_train=True, transform=None):
        self.samples = []  # Dictionary to hold class names and paths
        self.labels_map = {}  # Dictionary to hold class names and indices
        self.is_train = is_train
        self.transform = transform
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
                self.samples.append((waveform, idx_class, sr))



    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index][0], self.samples[index][1]







# class CustomDataset(Dataset):
#     def __init__(self, csv_path, audio_dir, transform=None):
#         self.df = pd.read_csv(csv_path)
#         self.audio_dir = audio_dir
#         self.transform = transform
#
#     def __len__(self):
#         return len(self.df)
#
#     def __getitem__(self, idx):
#         wav_path = os.path.join(self.audio_dir, self.df.iloc[idx, 0])
#         label = self.df.iloc[idx, 1]
#         # Load audio
#         audio, sample_rate = torchaudio.load(wav_path)
#         # Apply transforms
#         if self.transform:
#             audio = self.transform(audio)
#         return audio, label
