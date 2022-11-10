import custom_dataset
from model import M5
import torch
import torchaudio
import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import sounddevice as sd
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from playsound import playsound
import pyaudio


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
            axes[c].set_ylabel(f"Channel {c + 1}")
    figure.suptitle("waveform")
    plt.show(block=True)


def pad_sequence(batch):
    # Make all tensor in a batch the same length by padding with zeros
    batch = [item.t() for item in batch]
    batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0.)
    return batch.permute(0, 2, 1)

def collate_fn(batch):
    tensors, targets = [], []
    for waveform, idx_class, sr in batch:
        tensors += [waveform]
        targets += [torch.tensor(idx_class)]

        # Group the list of tensors into a batched tensor
    tensors = pad_sequence(tensors)
    targets = torch.stack(targets)

    return tensors, targets


def number_of_correct(pred, target):
    return pred.squeeze().eq(target).sum().item()

def padding(waveform, max_len):
    # Pad the waveform
    length_waveform = waveform.shape[1]
    if length_waveform < max_len:
        waveform = torch.cat((waveform, torch.zeros((1, max_len - length_waveform))), dim=1)
    return waveform


def get_likely_index(tensor):
    return tensor.argmax(dim=-1)

def train(model, device, train_loader, optimizer, epoch):

     for epoch in range(epoch):

        # Test
        model.eval()
        correct = 0
        with torch.no_grad():
            for batch_idx, (waveforms, targets) in enumerate(test_loader):
                waveforms = waveforms.to(device)
                targets = targets.to(device)
                output_ac = model(waveforms)
                pred = get_likely_index(output_ac)
                correct += number_of_correct(pred, targets)

        print("Test accuracy: ", correct / len(custom_dataset_test))

        loss_epoch = 0

        # Train
        model.train()
        for batch_idx, (waveforms, targets) in enumerate(train_loader):
            sample_count = len(waveforms)
            waveforms = waveforms.to(device)
            targets = targets.to(device)
            output_ac = model(waveforms)
            loss = F.nll_loss(output_ac.squeeze(), targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #print("Batch: ", batch_idx, " Sample count: ", sample_count, " Loss: ", loss.item())
            loss_epoch += loss.item()

        loss_epoch /= len(train_loader)
        losses_train_list.append(loss_epoch)
        print("\nEpoch: ", epoch, " Loss: ", loss_epoch)
        print("-----------------------------------------------------")



if __name__ == "__main__":
    # Create a custom dataset object
    path_dataset = "/home/goktug/projects/Ceres/dataset/"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sr_target = 8000
    max_length = 32000
    batch_size = 2
    transform = True

    custom_dataset_train = custom_dataset.CustomDataset(path_dataset, True, transform,
                                                        max_length, sr_target)


    train_loader = torch.utils.data.DataLoader(custom_dataset_train, batch_size=batch_size,
                                               shuffle=True, num_workers=1, collate_fn=collate_fn,
                                               pin_memory=True)

    custom_dataset_test = custom_dataset.CustomDataset(path_dataset, False, transform, max_length, sr_target)
    test_loader = torch.utils.data.DataLoader(custom_dataset_test, batch_size=batch_size,
                                              shuffle=True, num_workers=1, collate_fn=collate_fn,
                                              pin_memory=True)

    # Iterate over batches train:
    sample_waveform = None
    for idx, (waveforms, targets) in enumerate(train_loader):
        # print("Batch index: ", idx)
        # print("Waveforms shape: ", len(waveforms))
        # print("Targets: ", targets)
        sample_waveform = waveforms[0]
        break
        # plot_waveform(waveforms[0], sr_target)

    print("Number of samples in train dataset: ", len(custom_dataset_train))
    print("Number of samples in test dataset: ", len(custom_dataset_test))
    print("Sample rate: ", sr_target)
    print("Wavelength: ", len(sample_waveform[0]))

    print(custom_dataset_train.labels_map)
    print(custom_dataset_test.labels_map)


    model = M5(n_input=sample_waveform.shape[0],
               n_output=custom_dataset_train.getLabelCount())
    model.to(device)


    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    losses_train_list = []

    train(model, device, train_loader, optimizer, 1)


    path_ac = "/home/goktug/Downloads/ac.wav"
    waveform_ac, sr_ac = torchaudio.load(path_ac)
    waveform_ac = waveform_ac[0]
    waveform_ac = waveform_ac.reshape(1, -1)
    transform = torchaudio.transforms.Resample(orig_freq=sr_ac, new_freq=sr_target)
    waveform_ac = transform(waveform_ac)
    waveform_ac = padding(waveform_ac, max_length)
    print("Loaded audio with sample rate: ", sr_ac)
    print("shape: ", waveform_ac.shape)
    waveform_ac_np = waveform_ac.numpy()
    print(waveform_ac_np.shape)

    #plot_waveform(waveform_ac, sr_target)

    # instantiate PyAudio (1)
    p = pyaudio.PyAudio()

    # open stream (2), 2 is size in bytes of int16
    stream = p.open(format=p.get_format_from_width(2),
                    channels=1,
                    rate=sr_target,
                    output=True)

    # play stream (3), blocking call
    stream.write(waveform_ac_np)

    # stop stream (4)
    stream.stop_stream()
    stream.close()

    # close PyAudio (5)
    p.terminate()



    if transform:

        model.eval()

        with torch.no_grad():
            waveform_batch_ac = [waveform_ac, waveform_ac]
            waveform_batch_ac = pad_sequence(waveform_batch_ac)
            waveform_batch_ac = waveform_batch_ac.to(device)
            output_ac = model(waveform_batch_ac)
            pred_ac = get_likely_index(output_ac)
            print("Predicted class: ", pred_ac.t())

        print("----------------------------------------------")








