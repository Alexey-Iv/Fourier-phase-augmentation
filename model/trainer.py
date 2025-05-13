import torchvision.models as models
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.datasets import ImageFolder
import torchvision
from torchvision import datasets
from torch import nn
import torch.optim as optim
import torch
from torchvision.transforms import transforms
import numpy as np
from tqdm import tqdm
import random
import os
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt
import sys
from resnet import get_resnet
from dataset import norm_transform, Iris_Classification_Dataset
sys.path.append("../graphics")
from make_gif import create_animation_from_dataset
from apha import APFA


# this program for classification problem
## -------------------------------------------------------------- ##

PATH_DATA = "../Norm_photo"
NETWORK = True


class Trainer():
    def __init__(
        self,
        model,
        train_dl=None,
        test_dl=None,
        num_epochs=30,
        num_workers=4
    ):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_epochs = num_epochs
        self.e = 0

        self.model = model.to(self.device)
        self.train_dl = train_dl
        self.test_dl = test_dl

        # Инициализация списков для хранения истории обучения
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies_top1 = []
        self.val_accuracies_top5 = []

        self.best_acc = 0.0
        self.number = 0
        self.criterion = None

        current_time = datetime.now()
        formated_time = current_time.strftime("%m-%d_%H:%M:%S")
        self.dir_model = f"./models_out_{formated_time}"
        self.dir_plot = self.dir_model + "/plot"

        if not os.path.exists(self.dir_model):
            os.makedirs(self.dir_model)

        if not os.path.exists(self.dir_plot):
            os.makedirs(self.dir_plot)

    def train(self):
        criterion, optimizer = self._get_optimizer(self.model.parameters())
        scheduler = self._get_scheduler(optimizer)

        for e in range(self.num_epochs):
            self.model.train()

            self.e = e
            with tqdm(
                self.train_dl,
                desc=f"Epoch {e+1}/{self.num_epochs} [Train]",
                leave=False,
                dynamic_ncols=True
            ) as pbar:

                train_loss = 0.0
                for x_batch, y_batch in pbar:
                    x_batch = x_batch.to(self.device)
                    y_batch = y_batch.to(self.device)

                    optimizer.zero_grad()

                    logits = self.model(x_batch)
                    loss = criterion(logits, y_batch)

                    loss.backward()
                    optimizer.step()

                    train_loss += loss.item() * x_batch.size(0)

                    pbar.set_postfix({'loss': f"{loss.item():.4f}"})

            train_loss /= len(self.train_dl.dataset)
            self.train_losses.append(train_loss)

            # Output after each epoch
            print(f"Epoch {e+1}, Loss: {train_loss:.4f}")

            _, val_accuracy_top1, _ = self.valid()

            scheduler.step(val_accuracy_top1)

            self._save_schedule(val_accuracy_top1)

            print(f"LR после обновления: {optimizer.param_groups[0]['lr']}")

        torch.save(self.model.state_dict(), f"{self.dir_model}/model_super.pth")

        self.save_graphics()

    def valid(self, weights_model=None):
        self.model.eval()

        # Validation
        val_loss = 0.0
        correct_top1 = 0
        correct_top5 = 0
        total = 0

        e = self.e
        with tqdm(
            self.test_dl,
            desc=f"Epoch {e+1}/{self.num_epochs} [Val]",
            leave=False,
            dynamic_ncols=True
        ) as pbar:
            with torch.no_grad():
                for x_batch, y_batch in pbar:
                    x_batch = x_batch.to(self.device)
                    y_batch = y_batch.to(self.device)
                    #print(y_batch.shape)
                    outputs = self.model(x_batch)

                    loss = self.criterion(outputs, y_batch)

                    # top-1 accuracy
                    val_loss += loss.item() * x_batch.size(0)
                    _, predicted = torch.max(outputs.data, 1)
                    correct_top1 += (predicted == y_batch).sum().item()

                    # top-5 accuracy
                    _, predicted_top5 = torch.topk(outputs, k=5, dim=1)  # [batch_size, 5]
                    correct_top5 += torch.sum(predicted_top5 == y_batch.view(-1, 1)).item()

                    total += y_batch.size(0)

                    pbar.set_postfix({'acc_top_1': f"{100*correct_top1/total:.2f}%",
                                      'acc_top_5': f"{100*correct_top5/total:.2f}%"})

        val_loss /= len(self.test_dl.dataset)
        val_accuracy_top1 = 100 * correct_top1 / total # TOP_1
        val_accuracy_top5 = 100 * correct_top5 / total # TOP_5

        print(f"Val Loss: {val_loss:.4f}")
        print(f"Val Accuracy(TOP 1): {val_accuracy_top1:.2f}. \
            Val Accuracy(TOP 5): {val_accuracy_top5:.2f}%")

        self.val_losses.append(val_loss)
        self.val_accuracies_top1.append(val_accuracy_top1)
        self.val_accuracies_top5.append(val_accuracy_top5)

        return val_loss, val_accuracy_top1, val_accuracy_top5

    def _save_schedule(self, par):
        if par > self.best_acc:
            self.best_acc = par

            #self.save_model()

            self.number += 1

            print(f"Saved new best model with accuracy {par:.2f} with {self.number}%")

    def save_model(self):
        if self.model:
            torch.save(self.model.state_dict(), f"{self.dir_model}/model_{self.number}.pth")

    def save_graphics(self):
        sns.set_theme(style="whitegrid", context="talk", palette="colorblind")
        plt.rcParams['font.family'] = 'DejaVu Sans'

        # 1. Loss train/valid
        plt.figure(figsize=(13, 6))
        sns.lineplot(
            x=range(len(self.train_losses)),
            y=self.train_losses,
            label='Train Loss',
            linewidth=2.5,
            marker='o',
            markersize=8
        )
        sns.lineplot(
            x=range(len(self.val_losses)),
            y=self.val_losses,
            label='Val Loss',
            linewidth=2.5,
            marker='s',
            markersize=8
        )

        plt.title('Training and Validation Loss', fontsize=16, pad=20)
        plt.xlabel('Epoch', fontsize=14)
        plt.ylabel('Loss', fontsize=14)
        plt.legend(title='', frameon=True, facecolor='white')
        plt.tight_layout()
        plt.savefig(f"{self.dir_plot}/loss_output.svg", dpi=300, format="svg", bbox_inches='tight')
        plt.close()

        # 2. Accuracy
        plt.figure(figsize=(13, 6))
        palette = sns.color_palette("husl", 2)

        sns.lineplot(
            x=range(len(self.val_accuracies_top1)),
            y=self.val_accuracies_top1,
            label='TOP 1 ACC',
            color=palette[0],
            linewidth=2.5,
            marker='^',
            markersize=8
        )
        sns.lineplot(
            x=range(len(self.val_accuracies_top5)),
            y=self.val_accuracies_top5,
            label='TOP 5 ACC',
            color=palette[1],
            linewidth=2.5,
            marker='d',
            markersize=8
        )

        plt.title('Validation Accuracy', fontsize=16, pad=20)
        plt.xlabel('Epoch', fontsize=14)
        plt.ylabel('Accuracy (%)', fontsize=14)
        plt.legend(title='', frameon=True, facecolor='white')
        plt.tight_layout()
        plt.savefig(f"{self.dir_plot}/accuracy_output.svg", dpi=300, format="svg", bbox_inches='tight')
        plt.close()

        np.savez(f'{self.dir_plot}/array_losses.npz',
                train_losses=self.train_losses,
                val_losses=self.val_losses)

        np.savez(f'{self.dir_plot}/array_accuracy.npz',
                val_accuracies_top1=self.val_accuracies_top1,
                val_accuracies_top5=self.val_accuracies_top5)

    def _get_optimizer(
        self,
        parameters=None,
        learning_rate=1e-3,
        weight_decay=1e-4,
        label_smoothing=0.01
    ):
        criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.criterion = criterion
        optimizer = optim.AdamW(
            parameters,
            lr=learning_rate,
            weight_decay=weight_decay
        )

        return criterion, optimizer

    def _get_scheduler(self, optimizer):
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.1,
            patience=2
        )

        return scheduler


if __name__ == "__main__":
    train_dataset = Iris_Classification_Dataset(
        root_dir=PATH_DATA,
        train=True,
        transform=norm_transform
    )

    test_dataset = Iris_Classification_Dataset(
        root_dir=PATH_DATA,
        train=False,
        transform=norm_transform
    )

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)


    original_model_parameters ={
        'in_channels' : 2048,
        'num_parts' : 4,
        'gamma' : 0.05,
        'num_classes' : 512
    }

    if NETWORK:
        model = model = APFA(**original_model_parameters)
    else:
        model = get_resnet(original_model_parameters['num_classes'])


    tmp = Trainer(model, train_loader, test_loader)
    tmp.train()

    create_animation_from_dataset(20, test_dataset, "gif/iris_animation.gif")


