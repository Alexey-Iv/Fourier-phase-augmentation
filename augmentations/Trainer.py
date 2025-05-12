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
sys.path.append("../model/")

from resnet import get_resnet
from datasets import coll_fn_augm, Filtered_Dataset, train_transform, val_transform
import argparse
from dataset import Iris_Classification_Dataset


## -------------------------------------------------------------- ##

SEED = 42
torch.manual_seed(SEED)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class Trainer():
    def __init__(
        self,
        root=None,
        num_classes=200,
        batch_size=32,
        collate_fn=coll_fn_augm,
        train_transform=train_transform,
        valid_transform=valid_transform,
        num_epochs=30,
        num_workers=4
    ):
        self.root = root
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_classes = num_classes
        self.collate_fn = collate_fn
        self.train_transform = train_transform
        self.valid_transform = valid_transform
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.e = 0
        self.num_workers = num_workers

        self.model = None
        self.train_dl = None
        self.valid_dl = None

        # Инициализация списков для хранения истории обучения
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies_top1 = []
        self.val_accuracies_top5 = []

        self.best_acc = 0.0
        self.number = 0


        self.dir_model = f"./models_out_{datetime.now()}"
        self.dir_plot = self.dir_model + "/plot"

        if not os.path.exists(self.dir_model):
            os.makedirs(self.dir_model)

        if not os.path.exists(self.dir_plot):
            os.makedirs(self.dir_plot)

    def train(self):
        train_dataloader, _ = self._get_dataloaders()
        model = self._get_model()
        criterion, optimizer = self._get_optimizer()
        scheduler = self._get_scheduler()

        for e in range(self.num_epochs):
            model.train()

            self.e = e
            with tqdm(
                train_dataloader,
                desc=f"Epoch {e+1}/{num_epochs} [Train]",
                leave=False,
                dynamic_ncols=True
            ) as pbar:

                train_loss = 0.0
                for x_batch, y_batch in pbar:
                    x_batch = x_batch.to(self.device)
                    y_batch = y_batch.to(self.device)

                    optimizer.zero_grad()

                    logits = model(x_batch)
                    loss = criterion(logits, y_batch)

                    loss.backward()
                    optimizer.step()

                    train_loss += loss.item() * x_batch.size(0)

                    pbar.set_postfix({'loss': f"{loss.item():.4f}"})

            train_loss /= len(train_dataloader.dataset)
            self.train_losses.append(train_loss)

            # Output after each epoch
            print(f"Epoch {e+1}, Loss: {train_loss:.4f}")

            _, val_accuracies_top1, _ = valid()

            scheduler.step(val_accuracy_top1)

            _save_schedule(val_accuracies_top1)

            print(f"LR после обновления: {optimizer.param_groups[0]['lr']}")

        torch.save(model.state_dict(), f"{self.dir_model}/model_super.pth")

        save_graphics()

    def valid(self, weights_model=None):
        if self.model == None:
            pass
            # TODO
            # Add feature, whick allows to download the model's weights
        else:
            model = self.model

        if self.valid_dl == None:
            _, test_dataloader = self._get_dataloaders()
        else:
            test_dataloader = self.valid_dl


        model.eval()

        # Validation
        val_loss = 0.0
        correct_top1 = 0
        correct_top5 = 0
        total = 0

        e = self.e
        with tqdm(
            test_dataloader,
            desc=f"Epoch {e+1}/{self.num_epochs} [Val]",
            leave=False,
            dynamic_ncols=True
        ) as pbar:
            with torch.no_grad():
                for x_batch, y_batch in pbar:
                    x_batch = x_batch.to(self.device)
                    y_batch = y_batch.to(self.device)

                    outputs = model(x_batch)

                    loss = criterion(outputs, y_batch)

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

        val_loss /= len(test_dataloader.dataset)
        val_accuracy_top1 = 100 * correct_top1 / total # TOP_1
        val_accuracy_top5 = 100 * correct_top5 / total # TOP_5

        print(f"Val Loss: {val_loss:.4f}")
        print(f"Val Accuracy(TOP 1): {val_accuracy_top1:.2f}. \
            Val Accuracy(TOP 5): {val_accuracy_top5:.2f}%")

        self.val_losses.append(val_loss)
        self.val_accuracies_top1.append(val_accuracy_top1)
        self.val_accuracies_top5.append(val_accuracy_top5)

        return val_loss, val_accuracies_top1, val_accuracies_top5

    def _save_schedule(self, par):
        if par > self.best_acc:
            self.bes_acc = par

            save_model()

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

    def _get_model(self):
        model = get_resnet(self.num_classes).to(self.device)
        self.model = model

        return model

    def _get_optimizer(
        self,
        parameters=None,
        learning_rate=1e-3,
        weight_decay=1e-4,
        label_smoothing=0.01
    ):
        criterion = nn.CrossEntropy(label_smoothing=label_smoothing)
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

    def _get_datasets(self):
        # train_dataset = Filtered_Dataset(
        #     root=self.root,
        #     num_classes=self.num_classes,
        #     transform=self.train_transform,
        #     split='train',
        #     seed=42,
        #     test_size=0.2
        # )

        # Создаем val_dataset с теми же классами и разделением
        # val_dataset = Filtered_Dataset(
        #     root=self.root,
        #     num_classes=self.num_classes,
        #     transform=self.valid_transform,
        #     split='val',
        #     selected_classes=train_dataset.selected_classes,
        #     samples_split=train_dataset.samples_split
        # )

        train_dataset = Iris_Classification_Dataset(
            root=self.root,
            True,
            transform=self.train_transform
        )

        val_dataset = Iris_Classification_Dataset(
            root=self.root,
            False,
            transform=self.valid_transform
        )

        return train_dataset, val_dataset

    def _get_dataloaders(self):
        train_ds, val_ds = _get_datasets()

        train_dl = DataLoader(
            train_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            collate_fn=self.collate_fn
        )

        self.train_dl = train_dl

        val_dl = DataLoader(
            val_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

        self.valid_dl = valid_dl

        return train_dl, val_dl


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Get a result from working of phase augmentation.')
    parser.add_argument('root', type=str, help='Path to you directory of images.')
    parser.add_argument('num_cl', type=int, default=200, help='Count of classes.')
    args = parser.parse_args()

    model = Trainer(
        args.root,
        args.num_cl
    )

    model.train()

