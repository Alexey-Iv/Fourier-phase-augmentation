import torchvision.models as models
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.datasets import ImageFolder
import torchvision
import torch.nn.functional as F
from torch.nn.functional import normalize
from torchvision import datasets
from torch import nn
import torch.optim as optim
import torch
from torchvision.transforms import transforms
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
import os
from sklearn.manifold import TSNE


from model.apha import Triplet_Network
from metrics.metrics import calculate_eer, get_hist, scatter
from model.dataset import norm_transform, another_transform, IrisDataset, Triplet, get_embeddings, \
    get_dataloaders_to_IRIS, get_dl_2_IRIS, testing_model
from model.resnet import get_resnet, get_resnet152
from model.loss import TripletLoss, BatchHardTripletLoss, Hard_mining_TripletLoss


## -------------------------------------------------------------- ##


torch.manual_seed(42)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

NETWORK = True    # Do you want to use new Network? [True] or backbone (resnet) [False]
SUBTITLE = None

#hyper parameters:
num_epochs = 50
PATH_TO_NORMALIZE_PHOTOES = "./Norm_photo"
NUM_SEEN_CLASSES = 180
batch_size = 8
margin = 0.6
learning_rate = 0.0001


original_model_parameters = {
    'in_channels' : 2048,
    'num_parts' : 4,
    'gamma' : 0.05,
    'num_classes' : 512
}

# TODO переписать в lighting код

if __name__ == '__main__':

    if NETWORK:
        model = Triplet_Network(original_model_parameters).to(device)
        SUBTITLE = "Phase Model"
    else:
        model = get_resnet(original_model_parameters['num_classes']).to(device)
        SUBTITLE = "ResNet50"


    train_dataloader, few_dataloader, test_dataloader = get_dl_2_IRIS(
        PATH_TO_NORMALIZE_PHOTOES,
        NUM_SEEN_CLASSES,
        batch_size,
        norm_transform
    )

    criterion = Hard_mining_TripletLoss(margin=margin, device=device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)

    val_losses = []

    best_auc = 0.0
    train_losses = []
    val_metrics = {'FAR': [], 'FRR': [], 'ERR': [], 'AUC': []}
    eer_list = []
    val_accuracies = []

    print(f"Starts fiting the {SUBTITLE}, Margin = {margin}")
    for e in range(num_epochs):
        model = model.train()

        with tqdm(train_dataloader,
                desc=f"Epoch {e+1}/{num_epochs} [Train]",
                leave=False,
                dynamic_ncols=True
        ) as pbar:
            train_loss = 0.0
            for batch in pbar:
                images, labels = batch
                images = images.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                embeddings = model(images)

                loss = criterion(embeddings, labels)

                loss.backward()

                optimizer.step()

                train_loss += loss.item() * images.shape[0]

                pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        train_loss /= len(train_dataloader.dataset)
        train_losses.append(train_loss)

        print(f"Epoch {e+1}, Loss: {train_loss:.4f} \n LR: {optimizer.param_groups[0]['lr']}")

        model.eval()
        #print(f"Result : {check(resnet50, model, test_dataloader, device) * 100:.4f} %")
        acc, _, __ = testing_model(model, few_dataloader, test_dataloader, device=device)
        val_accuracies.append(acc)

        embeddings, labels = get_embeddings(model, test_dataloader, device=device)
        metrics = calculate_eer(embeddings, labels)

        eer_list.append(metrics['eer'])

        print(f"""
                EER: {metrics['eer']:.7f}
                Threshold: {metrics['threshold']:.4f}
                Genuine distances: {metrics['genuine_mean']:.2f} ± {metrics['genuine_std']:.2f}
                Impostor distances: {metrics['impostor_mean']:.2f} ± {metrics['impostor_std']:.2f}
        """)

        if e % 5 == 0:
            tsne = TSNE(random_state=0, perplexity=20)
            embeddings, labels = get_embeddings(model, test_dataloader, device=device)
            train_tsne_embeds = tsne.fit_transform(embeddings)

            get_hist(model, test_dataloader, device=device, out=f"hist_{SUBTITLE}_{e}.png", SUBTITLE)

            scatter(train_tsne_embeds, labels.astype(np.int32), f"{SUBTITLE}_{e}")


        scheduler.step(train_loss)

