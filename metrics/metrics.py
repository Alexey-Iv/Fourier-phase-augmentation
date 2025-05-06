from collections import defaultdict
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.metrics import roc_curve
import torch
from itertools import combinations
import matplotlib.pyplot as plt
from model.dataset import get_embeddings
import seaborn as sns
import matplotlib.patheffects as PathEffects
import numpy as np
import os
from datetime import datetime


def calculate_eer(embeddings, labels):
    unique_labels = np.unique(labels)
    genuine_dists = []
    impostor_dists = []

    for label in unique_labels:
        class_indices = np.where(labels == label)[0]
        class_emb = embeddings[class_indices]
        # 1. Внутриклассовые сравнения
        if len(class_indices) >= 2:
            #class_emb = embeddings[class_indices]
            dists = cdist(class_emb, class_emb, 'euclidean')
            genuine = dists[np.triu_indices(len(class_emb), 1)]  # Исправлено
            genuine_dists.extend(genuine)

        # 2. Межклассовые сравнения
        other_indices = np.where(labels != label)[0]
        if len(other_indices) > 0 and len(class_indices) > 0:
            other_emb = embeddings[other_indices]
            imp_dists = cdist(class_emb, other_emb, 'euclidean').flatten()
            impostor_dists.extend(imp_dists)

        # 5. Проверка на пустые списки
    if len(genuine_dists) == 0 or len(impostor_dists) == 0:
        return {
            'eer': 1.0,
            'threshold': None,
            'genuine_mean': 0,
            'impostor_mean': 0,
            'genuine_std': 0,
            'impostor_std': 0
        }

    # 3. Формирование меток
    y_true = np.concatenate([np.ones(len(genuine_dists)),
                            np.zeros(len(impostor_dists))])
    y_score = -np.concatenate([genuine_dists, impostor_dists])

    # 4. Расчет EER
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    fnr = 1 - tpr
    eer_idxx = np.nanargmin(np.abs(fnr - fpr))


    return {
        'eer': fpr[eer_idxx],
        'threshold': thresholds[eer_idxx],
        'genuine_mean': np.mean(genuine_dists),
        'impostor_mean': np.mean(impostor_dists),
        'genuine_std': np.std(genuine_dists),
        'impostor_std': np.std(impostor_dists)
    }


# Новая функция вычисления EER с учетом реального диапазона расстояний
def cal_eer(target, imposter):
    min_score = min(target.min(), imposter.min())
    max_score = max(target.max(), imposter.max())

    thresholds = torch.linspace(min_score, max_score, 1000)

    fars = torch.tensor([(imposter <= t).float().mean() for t in thresholds])
    frrs = torch.tensor([(target > t).float().mean() for t in thresholds])

    abs_diffs = torch.abs(fars - frrs)
    min_index = torch.argmin(abs_diffs)

    eer = (fars[min_index] + frrs[min_index]) / 2
    eer_threshold = thresholds[min_index]

    return eer, eer_threshold


def get_hist(model, test_dl, device=None, out="result.png", root=None):
    if root == None:
        root = f"hist_{datetime.now()}"
        if not os.path.exists(root):
            os.makedirs(root)

    embeddings, labels = get_embeddings(model, test_dl, device)

    # Создаем словарь для группировки
    class_embeddings = {}

    # Проходим по всем эмбеддингам и меткам
    for emb, label in zip(embeddings, labels):
        label = label.item()  # Если метки в тензоре

        if label not in class_embeddings:
            # Создаем новый ключ с добавлением размерности батча
            class_embeddings[label] = torch.tensor(emb).unsqueeze(0)
        else:
            # Конкатенируем с существующими эмбеддингами класса
            class_embeddings[label] = torch.cat([
                class_embeddings[label],
                torch.tensor(emb).unsqueeze(0)
            ], dim=0)

    embeddings = class_embeddings

    all_target_scores = []
    all_imposter_scores = []

    for class_id in embeddings:
        # Target-пары внутри класса
        class_embs = embeddings[class_id]
        if class_embs.shape[0] > 1:
            # Генерация всех уникальных пар
            indices = torch.tensor(list(combinations(range(class_embs.shape[0]), 2)))
            target_pairs_a = class_embs[indices[:, 0]]
            target_pairs_b = class_embs[indices[:, 1]]
            target_scores = torch.norm(target_pairs_a - target_pairs_b, p=2, dim=1)
            all_target_scores.append(target_scores)

        # Imposter-пары с другими классами
        for other_class_id in embeddings:
            if other_class_id != class_id:
                other_embs = embeddings[other_class_id]

                # Генерация всех возможных комбинаций
                class_indices = torch.arange(class_embs.shape[0])
                other_indices = torch.arange(other_embs.shape[0])

                # Декартово произведение индексов
                pairs = torch.cartesian_prod(class_indices, other_indices)

                # Выборка соответствующих эмбеддингов
                imposter_pairs_a = class_embs[pairs[:, 0]]
                imposter_pairs_b = other_embs[pairs[:, 1]]

                imposter_scores = torch.norm(imposter_pairs_a - imposter_pairs_b, p=2, dim=1)
                all_imposter_scores.append(imposter_scores)

    # Объединение результатов
    target_scores = torch.cat(all_target_scores) if all_target_scores else torch.tensor([])
    imposter_scores = torch.cat(all_imposter_scores)


    # Гистограммы

    # Пересчет и построение графика с новой EER-точкой
    eer, eer_threshold = cal_eer(target_scores, imposter_scores)

    plt.figure(figsize=(10, 6))

    total_samples_target = target_scores.size()
    total_samples_imposter = imposter_scores.size()

    # Построение гистограммы с вероятностями
    plt.hist(target_scores.numpy(), bins=100, alpha=0.5, label='Target',
            weights=np.ones_like(target_scores.numpy()) / total_samples_target)  # Нормализация


    plt.hist(imposter_scores.numpy(), bins=100, alpha=0.5, label='Imposter',
             weights=np.ones_like(imposter_scores.numpy()) / total_samples_imposter)

    plt.axvline(x=eer_threshold.item(), color='red', linestyle='--', linewidth=2, label=f'EER Threshold ({eer_threshold:.2f})')
    plt.xlabel('Score (Euclidean)')
    plt.ylabel('Probability')
    plt.legend()

    plt.savefig(os.path.join(root, out))

    #plt.show()
    plt.close()

    print(f"EER: {eer.item():.4f} at threshold: {eer_threshold.item():.4f}")


# Define our own plot function
def scatter(x, labels, root='plot', subtitle=None):
    labels -= 240
    num_classes = len(set(labels)) # Calculate the number of classes
    palette = np.array(sns.color_palette("hls", num_classes)) # Choosing color
    # Create a seaborn scatter plot #
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40,
                    c=palette[labels.astype(np.int32)])

    plt.xlim(-25, 25)
    plt.ylim(-25, 25)

    ax.axis('off')
    ax.axis('tight')

    # Add label on top of each cluster ##
    idx2name = [str(x+1) for x in range(num_classes)]
    txts = []
    for i in range(num_classes):
        # Position of each label.
        xtext, ytext = np.median(x[labels == i, :], axis=0)
        txt = ax.text(xtext, ytext, idx2name[i], fontsize=24)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)


    if subtitle != None:
        plt.suptitle(subtitle)

    if not os.path.exists(root):
        os.makedirs(root)

    plt.savefig(os.path.join(root, str(subtitle)))
    plt.close()
