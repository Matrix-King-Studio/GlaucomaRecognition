import os
import time
import torch
import numpy as np
import matplotlib.pyplot as plt

from copy import deepcopy

from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

from models.LeNet import LeNet


def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    best_acc = 0.0
    since = time.time()
    best_model_wts = deepcopy(model.state_dict())

    epoch_losses = []
    epoch_acces = []

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')

        # 每个 epoch 都有一个训练和验证阶段
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # 训练模式
            else:
                model.eval()  # 验证模式

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloader[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                # 前向传播
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, predicts = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # 反向传播与优化
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(predicts == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            epoch_losses.append(epoch_loss)
            epoch_acces.append(epoch_acc)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = deepcopy(model.state_dict())
        torch.save(model, f'weights/model-{epoch + 1}.pkl')
        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, epoch_losses, epoch_acces


def imshow(inp, title=None):
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloader['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, predicts = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images // 2, 2, images_so_far)
                ax.axis('off')
                ax.set_title(
                    f"predicted: {class_names[predicts[j]]}, targets={'Non-Glaucoma' if labels[j] == 1 else 'Glaucoma'}")
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)


def train(model, num_epochs):
    model = model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer_ft = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    model, epoch_losses, epoch_accuracies = train_model(model, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=num_epochs)

    plt.figure(figsize=(10, 5))
    plt.title("Loss And Accuracy")
    plt.plot(epoch_losses, label="loss")
    plt.plot(epoch_accuracies, label="accuracy")
    plt.xlabel("Iterations")
    plt.ylabel("Loss And Accuracy")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    # 数据增强和标准化
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    data_dir = r'E:\Glaucoma\Datasets\Training400'
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
    dataloader = {x: DataLoader(image_datasets[x], batch_size=4, shuffle=True, num_workers=4) for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes

    device = torch.device("cpu")

    model = LeNet()
    train(model, 2)
    visualize_model(model)
    plt.ioff()
    plt.show()
