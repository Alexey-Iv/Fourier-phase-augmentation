from collections import defaultdict
import numpy as np
from scipy.spatial.distance import cdist


def calculate_eer(embeddings, labels):
    unique_labels = np.unique(labels)
    genuine_dists = []
    impostor_dists = []

    for label in unique_labels:
        class_indices = np.where(labels == label)[0]

        # 1. Внутриклассовые сравнения
        if len(class_indices) >= 2:
            class_emb = embeddings[class_indices]
            dists = cdist(class_emb, class_emb, 'euclidean')
            genuine = dists[np.triu_indices(len(class_emb), 1)]  # Исправлено
            genuine_dists.extend(genuine)

        # 2. Межклассовые сравнения
        other_indices = np.where(labels != label)[0]
        if len(other_indices) > 0 and len(class_indices) > 0:
            other_emb = embeddings[other_indices]
            imp_dists = cdist(class_emb, other_emb, 'euclidean').flatten()
            impostor_dists.extend(imp_dists)

    # 3. Формирование меток
    y_true = np.concatenate([np.ones(len(genuine_dists)),
                            np.zeros(len(impostor_dists))])
    y_score = np.concatenate([genuine_dists, impostor_dists])

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
