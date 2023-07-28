from torchvision import datasets
from torch.utils.data import DataLoader, Dataset
import numpy as np

class CIFAR10Dataset(Dataset):
    def __init__(self, dataset, transforms) -> None:
        self.dataset = dataset
        self.transforms = transforms

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        image, label = self.dataset[index]
        image = np.array(image)

        if self.transforms is not None:
            image = self.transforms(image=image)["image"]
        
        return (image, label)
        


def get_data_loader(train_transforms, test_transforms, batch_size=64):
    """Gets instance of train and test loader of CIFAR 10 Dataset"""
    
    train = datasets.CIFAR10('./data', train=True, download=True)
    test = datasets.CIFAR10('./data', train=False, download=True)

    train_dataset = CIFAR10Dataset(train, train_transforms)
    test_dataset = CIFAR10Dataset(test, test_transforms)

    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, shuffle=True, batch_size=batch_size)

    return train_loader, test_loader