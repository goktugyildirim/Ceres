
import custom_dataset
import torch
import torchaudio

if __name__ == "__main__":
    # Create a custom dataset object
    path_dataset = "/home/goktug/projects/Ceres/dataset/"

    custom_dataset_train = custom_dataset.CustomDataset(path_dataset, True)
    train_loader = torch.utils.data.DataLoader(custom_dataset_train, batch_size=1, shuffle=True, num_workers=0)

    custom_dataset_test = custom_dataset.CustomDataset(path_dataset, False)
    test_loader = torch.utils.data.DataLoader(custom_dataset_test, batch_size=1, shuffle=True, num_workers=0)

    print("Number of samples in train dataset: ", len(custom_dataset_train))
    print("Number of samples in test dataset: ", len(custom_dataset_test))



