import torchvision.models as models
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.datasets import ImageFolder
import torchvision
import torch.nn.functional as F
from torch.nn.functional import cosine_similarity, normalize
from torchvision import datasets
from torch import nn
import torch.optim as optim
import torch
from torchvision.transforms import transforms


from model.apha import APFA
from metrics.metrics import calculate_eer
from model.dataset import norm_transform, IrisDataset, Triplet
from model.loss import TripletLoss

import numpy as np
import tqdm
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
import os
from PIL import Image
import itertools
from sklearn.manifold import TSNE


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#hyper parameters:
num_epochs = 50
PATH_TO_NORMALIZE_PHOTOES = "./Norm_photo"
NUM_SEEN_CLASSES = 1
batch_size = 16
margin = 0.2
learning_rate = 0.0001


original_model_parameters = {
    'in_channels' : 2048,
    'num_parts' : 6,
    'gamma' : 0.05,
    'num_classes' : 512
}


class Network(nn.Module):
    def __init__(self):
        super().__init__()

        # Ветвь для нормализованных изображений
        self.orig_branch = APFA(**original_model_parameters)

    def forward(self, orig_img):
        norm_emb = normalize(self.orig_branch(orig_img), p=2, dim=1)

        return norm_emb

def get_resnet(num_classes=512):
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    return model


def get_embeddings(model, dataloader):
    embeddings = []
    labels = []
    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            outputs = model(images)
            outputs = F.normalize(outputs)
            embeddings.append(outputs.cpu().numpy())
            labels.append(targets.cpu().numpy())
    return np.concatenate(embeddings), np.concatenate(labels)

if __name__ == '__main__':

    train_data = IrisDataset(
        PATH_TO_NORMALIZE_PHOTOES,
        NUM_SEEN_CLASSES,
        norm_transform,
        "train"
    )


    model = Network().to(device)
    #model = get_resnet().to(device)

    test_data_index = IrisDataset(
        PATH_TO_NORMALIZE_PHOTOES,
        NUM_SEEN_CLASSES,
        norm_transform,
        "test_few"
    )

    test_data = IrisDataset(
        PATH_TO_NORMALIZE_PHOTOES,
        NUM_SEEN_CLASSES,
        norm_transform,
        "test_all"
    )

    oneshot_dl = DataLoader(
        test_data_index,
        batch_size,
        num_workers=4,
        pin_memory=True,
    )

    test_dl = DataLoader(
        test_data,
        batch_size,
        num_workers=4,
        pin_memory=True,
    )

    train_data = Triplet(train_data)
    triplet_train_dl = DataLoader(
        train_data,
        batch_size,
        num_workers=4,
        pin_memory=True,
        shuffle=True,
    )

    train_dataloader = triplet_train_dl


    criterion = TripletLoss(margin=margin, device=device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)

    val_losses = []

    best_auc = 0.0
    train_losses = []
    val_metrics = {'FAR': [], 'FRR': [], 'ERR': [], 'AUC': []}

    for e in range(num_epochs):
        model = model.train()

        with tqdm(train_dataloader,
                desc=f"Epoch {e+1}/{num_epochs} [Train]",
                leave=False,
                dynamic_ncols=True
        ) as pbar:
            train_loss = 0.0
            for batch in pbar:
                xs, ys = batch

                fs = [model(x.to(device)) for x in xs]

                optimizer.zero_grad()

                loss = criterion(fs, ys)

                loss.backward()
                optimizer.step()

                train_loss += loss.item() * xs[0].shape[0]

                pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        train_loss /= len(train_dataloader.dataset)
        train_losses.append(train_loss)

        print(f"Epoch {e+1}, Loss: {train_loss:.4f} \n LR: {optimizer.param_groups[0]['lr']}")

        model.eval()
        embeddings, labels = get_embeddings(model, test_dl)

        metrics = calculate_eer(embeddings, labels)

        print(f"""
                EER: {metrics['eer']:.4f}
                Threshold: {metrics['threshold']:.4f}
                Genuine distances: {metrics['genuine_mean']:.2f} ± {metrics['genuine_std']:.2f}
                Impostor distances: {metrics['impostor_mean']:.2f} ± {metrics['impostor_std']:.2f}
        """)

        if (e+1) % 5 == 0:
            tsne = TSNE(n_components=2, perplexity=30)
            emb_2d = tsne.fit_transform(embeddings[:500])  # Первые 500 примеров

            plt.figure(figsize=(10,8))
            plt.scatter(emb_2d[:,0], emb_2d[:,1], c=labels[:500], cmap='jet', alpha=0.6)
            plt.title(f't-SNE Visualization (Epoch {e+1})')
            plt.savefig(f'tsne_epoch_{e+1}.png')
            plt.close()
        scheduler.step(train_loss)
