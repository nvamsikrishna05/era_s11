import os
import torch
from tqdm import tqdm

from dataset import get_data_loader
from utils import get_transforms, get_device
from models import ResNet18
import matplotlib.pyplot as plt

# Training
def train(trainloader, net, optimizer, criterion, device):
    pbar = tqdm(trainloader)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(pbar):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        pbar.set_description(desc=f'Training: Loss={loss.item()} Batch_id={batch_idx} Accuracy={100*correct/total:0.2f}')
    return train_loss

# Testing
def test(testloader, net, criterion, device):
    global best_acc
    net.eval()
    pbar = tqdm(testloader)
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            pbar.set_description(desc=f'Test: Loss={loss.item()} Batch_id={batch_idx} Accuracy={100*correct/total:0.2f}')
    return test_loss

def plot_model_performance(epochs, train_loss, test_loss):
    plt.plot(range(epochs), train_loss, 'g', label='Training loss')
    plt.plot(range(epochs), test_loss, 'b', label='Testing loss')
    plt.title('Training and Testing loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

def run(model, criterion, optimizer, train_loader, test_loader, max_epochs):
    torch.manual_seed(123)

    device = get_device()
    net = model.to(device)
    train_loss = []
    test_loss = []
    for epoch in range(0, max_epochs):
        print(f"Epoch: {epoch}")
        tr_loss = train(train_loader, net, optimizer, criterion, device)
        te_loss = test(test_loader, net, criterion, device)
        train_loss.append(tr_loss)
        test_loss.append(te_loss)
    
    return net, train_loss, test_loss

