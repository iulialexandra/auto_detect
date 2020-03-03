import argparse
import os
import re
import numpy as np
import torch
import torchvision
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy import misc
import cv2
from images_to_video import extract_subdirs, sort_by_digits
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torchvision.utils import make_grid
from torchvision import transforms, datasets
from PIL import Image


def load_images(class_path):
    class_subdirs = extract_subdirs(class_path, "class")
    sorted_dirs = sort_by_digits(class_subdirs)
    data_list = []
    labels_list = []
    for class_idx, class_dir in enumerate(sorted_dirs):
        class_images = []
        for image_path in os.listdir(class_dir):
            if ".png" in image_path:
                image = misc.imread(os.path.join(class_dir, image_path))
                class_images.append(cv2.resize(image, (64, 64)))
        data_list.append(class_images)
        labels_list.append([class_idx] * len(class_images))
    min_samples_num = min([len(l) for l in data_list])
    equalized_data_list = [l[:min_samples_num] for l in data_list]
    equalized_labels_list = [l[:min_samples_num] for l in labels_list]
    images = np.array(equalized_data_list)
    labels = np.array(equalized_labels_list)
    train_length = int(0.7 * min_samples_num)
    indices = np.arange(min_samples_num)
    np.random.shuffle(indices)
    train_indices = indices[:train_length]
    test_indices = indices[train_length:]

    train_ims = images[:, train_indices, :, :, :]
    train_targets = labels[:, train_indices]
    test_ims = images[:, test_indices, :, :, :]
    test_targets = labels[:, test_indices]

    train_ims = np.reshape(train_ims, (-1, 64, 64, 3))
    train_targets = np.reshape(train_targets, (-1))
    test_ims = np.reshape(test_ims, (-1, 64, 64, 3))
    test_targets = np.reshape(test_targets, (-1))
    return train_ims, train_targets, test_ims, test_targets


class AutoObjectDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.transform = transform
        self.ims = images
        self.labs = labels

    def __len__(self):
        im_shape = np.shape(self.ims)
        return im_shape[0]

    def __getitem__(self, idx):
        x = self.ims[idx]
        y = self.labs[idx]
        if self.transform:
            x = Image.fromarray(x.astype(np.uint8))
            x = self.transform(x)
        return x, y


class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()

        self.cnn1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=1, padding=2)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.cnn2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(32 * 16 * 16, 10)

    def forward(self, x):
        out = self.cnn1(x)
        out = self.relu1(out)
        out = self.maxpool1(out)
        out = self.cnn2(out)
        out = self.relu2(out)
        out = self.maxpool2(out)
        out = out.view(-1, self.num_flat_features(out))
        out = self.fc1(out)
        return out

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


def plot_images(loader):
    for i_batch, sample_batched in enumerate(loader):
        images = sample_batched[0]
        labels = sample_batched[1]

        imgrid = make_grid(images).numpy()
        plt.imshow(np.transpose(imgrid, (1, 2, 0)))
        plt.title(str(labels))
        plt.show()

        # detached_ims = images.cpu().detach().numpy()
        # detached_labs = labels.cpu().detach().numpy()

        if i_batch == 0:
            plt.show()
            break


def parse_args():
    """Parses arguments specified on the command-line
    """
    argparser = argparse.ArgumentParser('Train and evaluate Roshambo iCarl')
    argparser.add_argument('--image_dims',
                           help="the dimensions of the images we are working with",
                           default=(64, 64, 3))
    return argparser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    data_transforms = transforms.Compose([transforms.ColorJitter(brightness=0.2, contrast=0.2,
                                                                 saturation=0.1, hue=0.07),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          ])
    train_ims, train_targets, test_ims, test_targets = load_images(
        "/mnt/Storage/code/object detection/auto_collected_data/TLP/images")
    train_dataset = AutoObjectDataset(train_ims, train_targets, transform=data_transforms)
    test_dataset = AutoObjectDataset(test_ims, test_targets, transform=data_transforms)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    path = "/mnt/Storage/code/object detection/auto_detect/results/net.pth"

    model = CNNModel()
    print(model)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(100):
        model.train()
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data in test_loader:
                images, labels = data[0].to(device), data[1].to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print("Accuracy at epoch {}: {} %".format(epoch, 100*correct/total))

        torch.save(model.state_dict(), path)

    model = CNNModel()
    model.load_state_dict(torch.load(path))
    model.to(device)
    model.eval()
    correct = 0
    total = 0
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    with torch.no_grad():
        for data in test_loader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            c = (predicted == labels).squeeze()
            correct += c.sum().item()
            for i, lab in enumerate(labels):
                class_correct[lab] += c[i].item()
                class_total[lab] += 1
    print("Accuracy at end of training: {} %".format(100*correct/total))
    for i in range(10):
        print("Accuracy of class {}: {} %".format(i, 100 * class_correct[i]/class_total[i]))


if __name__ == '__main__':
    main()
