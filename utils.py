import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from PIL import Image
import numpy as np
from torchsummary import summary

def get_device():
    """Gets the Device Type available on the machine"""
    if torch.cuda.is_available():
        print(f"Device Type - cuda")
        return torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        print(f"Device Type - mps")
        return torch.device("mps")
    else:
        print(f"Device Type - cpu")
        return torch.device("cpu")

def print_model_summary(model, input_size = (3, 32, 32)):
    """Prints the model summary"""
    summary(model, input_size)

def get_transforms():
    """Gets instance of train and test transforms"""
    mean = (0.4915, 0.4823, .4468)

    train_transform = A.Compose([
        A.Normalize(mean=mean, std=(0.2470, 0.2435, 0.2616), always_apply=True),
        A.PadIfNeeded(min_height=40, min_width=40, always_apply=True),
        A.RandomCrop(height=32, width=32, always_apply=True),
        A.CoarseDropout(max_holes=1, max_height=8, max_width=8, min_holes=1, fill_value=mean, mask_fill_value=None),
        ToTensorV2()
    ])

    test_transform = A.Compose([
        A.Normalize(mean=mean, std=(0.2470, 0.2435, 0.2616), always_apply=True),
        ToTensorV2()
    ])

    return train_transform, test_transform

def get_incorrrect_predictions(model, loader):
    """ Gets the incorrect prections """
    model.eval()
    device = get_device()
    incorrect = []
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            for d, t, p, o in zip(data, target, pred, output):
                if p.eq(t.view_as(p)).item() == False:
                    incorrect.append(
                        [d.cpu(), t.cpu(), p.cpu(), o[p.item()].cpu()])

    return incorrect

def plot_incorrect_predictions(predictions, class_map, title="Incorrect Predictions", count=10):
    """ Plots Incorrect predictions """
    print(f'Total Incorrect Predictions {len(predictions)}')

    if not count % 5 == 0:
        print("Count should be multiple of 10")
        return

    fig = plt.figure(figsize=(10, 5))
    plt.title(title)
    for i, (d, t, p, o) in enumerate(predictions):
        ax = fig.add_subplot(int(count/5), 5, i + 1, xticks=[], yticks=[])
        ax.set_title(f'{t}/{p}')
        plt.imshow(d.cpu().numpy().transpose(1, 2, 0))
        if i+1 == 5*(count/5):
            break

def grad_cam_image(model, target_layers, input_img):
    min = np.min(input_img.numpy())
    max = np.max(input_img.numpy())
    input_image = (input_img - min) / (max - min)
    model = model.to('cpu')
    target_layers = [model.layer3[-1]]
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False)
    input_tensor =  input_image.unsqueeze(0)
    grayscale_cam = cam(input_tensor=input_tensor)
    grayscale_cam = grayscale_cam[0, :]
    visualization = show_cam_on_image(input_image.numpy().transpose(1, 2, 0), grayscale_cam, use_rgb=True)
    return visualization

def show_gradcam(model, target_layers, predictions, N = 12):
    images = []
    for idx, data in enumerate(predictions):
        if idx >= N:
            break
        input_img = data[0]
        images.append(grad_cam_image(model, target_layers, input_img))
    print(len(images))
    fig = plt.figure(figsize=(10, 10))
    rows, cols = 4, 3
    for i in range(0, rows * cols):
        fig.add_subplot(rows, cols, i+1)
        plt.imshow(images[i])
    plt.show()

        
        
