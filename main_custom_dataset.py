
import custom_dataset
import torch
import torchaudio
import torch
import matplotlib.pyplot as plt


def plot_waveform(waveform, sample_rate):
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sample_rate

    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].plot(time_axis, waveform[c], linewidth=1)
        axes[c].grid(True)
        if num_channels > 1:
            axes[c].set_ylabel(f"Channel {c+1}")
    figure.suptitle("waveform")
    plt.show(block=True)



def collate_fn(batch):
    tensors, targets = [], []
    for waveform, class_name, sr in batch:
        tensors += [waveform]
        targets += [class_name]

    return tensors, targets



if __name__ == "__main__":
    # Create a custom dataset object
    path_dataset = "/home/goktug/projects/Ceres/dataset/"

    sr_target = 8000
    max_length = 16000
    batch_size = 64
    transform = False

    custom_dataset_train = custom_dataset.CustomDataset(path_dataset, True, transform,
                                                        max_length, sr_target)
    # for idx in range(len(custom_dataset_train)):
    #     waveform, class_name, sr = custom_dataset_train[idx]
    #     print("Waveform shape: ", waveform.shape)
    #     print("Class name: ", class_name)

    train_loader = torch.utils.data.DataLoader(custom_dataset_train, batch_size=batch_size,
                                               shuffle=True, num_workers=1, collate_fn=collate_fn,
                                               pin_memory=True)

    custom_dataset_test = custom_dataset.CustomDataset(path_dataset, True, transform, max_length, sr_target)
    test_loader = torch.utils.data.DataLoader(custom_dataset_test, batch_size=batch_size,
                                              shuffle=True, num_workers=1, collate_fn=collate_fn,
                                              pin_memory=True)

    print("Number of samples in train dataset: ", len(custom_dataset_train))
    print("Number of samples in test dataset: ", len(custom_dataset_test))
    print("Sample rate: ", sr_target)

    # Iterate over batches train:
    for idx, (waveforms, targets) in enumerate(train_loader):
        print("Batch index: ", idx)
        print("Waveforms shape: ", len(waveforms))
        print("Targets: ", targets)
        plot_waveform(waveforms[0], sr_target)
        break









