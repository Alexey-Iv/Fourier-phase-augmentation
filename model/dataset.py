from torchvision.transforms import transforms
import torch
from collections import defaultdict
import glob
import os
from PIL import Image
import random
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.model_selection import train_test_split


def coll_fn_augm(batch):
    batch_size = len(batch)
    images, labels = zip(*batch)
    batch = torch.tensor(images)

    alpha = 0.1

    h_freq = torch.fft.fftshift(torch.fft.fft2(batch, norm='ortho'))
    amplitude = torch.abs(h_freq)

    phase_origin = torch.angle(h_freq)

    h_ran = h_freq[random.randint(0, batch_size-1)] # [channels, H, W]
    phase_random = torch.angle(h_ran)
    phase_new = alpha * phase_random.unsqueeze(0) + (1 - alpha) * phase_origin
    h_freq_new = amplitude * torch.exp(1j * phase_new)

    output = torch.fft.ifft2(torch.fft.ifftshift(h_freq_new), norm='ortho').real

    return output


#(64, 512, 3)
norm_transform = transforms.Compose([
    transforms.Resize((64, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

another_transform = transforms.Compose([
    # Геометрические преобразования (применяются к исходному изображению до изменения размера)
    transforms.RandomApply([
        transforms.RandomRotation(10),  # Случайный поворот ±10 градусов
        transforms.RandomPerspective(  # Перспективные искажения
            distortion_scale=0.15,
            p=0.3
        ),
    ], p=0.5),

    # Цветовые преобразования (важно для разных условий освещения)
    transforms.ColorJitter(
        brightness=0.15,  # Яркость
        contrast=0.15,    # Контраст
        saturation=0.1,   # Насыщенность
        hue=0.05          # Оттенок (малое значение для сохранения цветов радужки)
    ),

    # Размытия и шумы
    transforms.RandomApply([
        transforms.GaussianBlur(  # Гауссово размытие
            kernel_size=3,
            sigma=(0.1, 1.5))
    ], p=0.3),

    transforms.Resize((224, 224)),
    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


class Iris_Classification_Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root_dir,
        train=True,
        transform=None,
        test_size=0.2,
        random_state=42
):
        self.root_dir = root_dir
        self.transform = transform
        self.train = train

        image_paths = []
        labels = []

        # Проходим по всем папкам пациентов
        patient_folders = [d for d in os.listdir(root_dir)
                          if os.path.isdir(os.path.join(root_dir, d)) and d.isdigit()]

        for patient_folder in sorted(patient_folders):
            patient_id = int(patient_folder)
            patient_path = os.path.join(root_dir, patient_folder)

            # Обрабатываем обе стороны глаза
            for side in ['L', 'R']:
                side_path = os.path.join(patient_path, side)

                if not os.path.exists(side_path):
                    continue

                # Собираем изображения с поддержкой форматов
                for img_name in os.listdir(side_path):
                    if len(os.listdir(side_path)) == 1:
                        continue

                    img_path = os.path.join(side_path, img_name)
                    image_paths.append(img_path)

                    # Формируем метку класса
                    class_label = (patient_id - 1) * 2 + (0 if side == 'L' else 1)

                    labels.append(class_label)


        # Стратифицированное разделение данных
        train_idx, test_idx = train_test_split(
            range(len(image_paths)),
            test_size=test_size,
            random_state=random_state,
            stratify=labels,
            shuffle=True
        )

        # Выбираем соответствующую выборку
        if self.train:
            self.image_paths = [image_paths[i] for i in train_idx]
            self.labels = [labels[i] for i in train_idx]
        else:
            self.image_paths = [image_paths[i] for i in test_idx]
            self.labels = [labels[i] for i in test_idx]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label


class IrisDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root=None,
        num_seen_classes=20,
        transform=None,
        mode=None,
        list_files=None
    ):
        self.root = root
        self.transform = transform
        self.mode = mode

        self.class_to_idxs = defaultdict(list)
        self.data = []

        all_patients = sorted(
            glob.glob(os.path.join(self.root, '*')),
            key=lambda x: int(os.path.basename(x))
        )

        if mode == "train":
            self.patients = all_patients[:num_seen_classes]
            self.need_classes = list(range(num_seen_classes * 2))

            for patient_dir in self.patients:
                patient_id = int(os.path.basename(patient_dir))
                for eye_dir in ['L', 'R']:
                    eye_path = os.path.join(patient_dir, eye_dir)
                    if os.path.exists(eye_path):
                        images = glob.glob(os.path.join(eye_path, '*.*'))
                        eye_class = 2 * patient_id if eye_dir == 'L' else 2 * patient_id + 1
                        self.data.extend([(img, eye_class) for img in images])

        elif mode in ["test_few", "test_all"]:
            self.patients = all_patients[num_seen_classes:]
            total_patients = len(all_patients)
            self.need_classes = [2*i for i in range(num_seen_classes, total_patients)]

            for patient_dir in self.patients:
                patient_id = int(os.path.basename(patient_dir))
                for eye_dir in ['L', 'R']:
                    eye_path = os.path.join(patient_dir, eye_dir)
                    if os.path.exists(eye_path):
                        images = glob.glob(os.path.join(eye_path, '*.*'))
                        eye_class = 2 * patient_id if eye_dir == 'L' else 2 * patient_id + 1

                        if mode == "test_few":
                            selected_images = images[:1]
                        else:
                            selected_images = images[1:]

                        self.data.extend([(img, eye_class) for img in selected_images])

        else:
            raise ValueError("Invalid mode. Use 'train' or 'test'")

        # Обновляем индексы классов
        for idx, (_, label) in enumerate(self.data):
            self.class_to_idxs[label].append(idx)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_path, label = self.data[index]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)
        return image, label



def random_choice_except(options, exception):
    choice = exception
    while choice == exception:
        choice = random.choice(options)
    return choice


class Triplet(torch.utils.data.Dataset):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset

    def __getitem__(self, index):
        sample1, target1 = self.dataset[index]

        positive_index = random_choice_except(
            self.dataset.class_to_idxs[target1],
            exception=index,
        )
        sample2, target2 = self.dataset[positive_index]

        negative_target = random_choice_except(
            self.dataset.need_classes,
            exception=target1,
        )

        negative_index = random.choice(
            self.dataset.class_to_idxs[negative_target],
        )

        sample3, target3 = self.dataset[negative_index]

        return [sample1, sample2, sample3], [target1, target2, target3]

    def __len__(self):
        return len(self.dataset)


def get_dataloaders_to_IRIS(
        path=None,
        num_seen=1,
        batch_size=1,
        transform_train=None,
        transform_test=None
    ):

    if transform_test == None:
        transform_test = transform_train

    train_data = IrisDataset(
        path,
        num_seen,
        transform_train,
        "train"
    )

    test_data = IrisDataset(
        path,
        num_seen,
        transform_test,
        "test_all"
    )

    test_dl = DataLoader(
        test_data,
        batch_size,
        num_workers=4,
        pin_memory=True,
    )

    train_dl = DataLoader(
        train_data,
        batch_size,
        num_workers=4,
        pin_memory=True,
        shuffle=True,
    )

    return train_dl, test_dl


def get_dl_2_IRIS(
    path=None,
    num_seen=1,
    batch_size=1,
    transform_train=None,
    transform_test=None
):
    if transform_test == None:
         transform_test = transform_train

    train_data = IrisDataset(
        path,
        num_seen,
        transform_train,
        "train"
    )

    test_data_few = IrisDataset(
        path,
        num_seen,
        transform_train,
        "test_few"
    )

    test_data_all = IrisDataset(
        path,
        num_seen,
        transform_train,
        "test_all"
    )

    test_dl_few = DataLoader(
        test_data_few,
        batch_size,
        num_workers=4,
        pin_memory=True,
    )

    test_dl_all = DataLoader(
        test_data_all,
        batch_size,
        num_workers=4,
        pin_memory=True,
    )

    train_dl = DataLoader(
        train_data,
        batch_size,
        num_workers=4,
        pin_memory=True,
        shuffle=True
    )

    return train_dl, test_dl_few, test_dl_all


def run_emb_net(emb_net, dataloader, device=None, normalize=False):
    data_x = []
    data_y = []
    device = device
    with torch.no_grad():
        for inputs, labels in dataloader:
            feats = emb_net(inputs.to(device))
            if normalize:
                feats = F.normalize(feats)
            feats = feats.detach().cpu().numpy()
            labels = labels.detach().cpu().numpy()
            data_x.append(feats)
            data_y.append(labels)

    data_x = np.concatenate(data_x, axis=0)
    data_y = np.concatenate(data_y, axis=0)
    return data_x, data_y


def train_knn(emb_net, oneshot_dl, device, normalize=False):
    data_x, data_y = run_emb_net(emb_net, oneshot_dl, device, normalize)
    knn = KNeighborsClassifier(n_neighbors=1)
    knn = knn.fit(data_x, data_y)
    return knn


def testing_model(emb_net, test_dl_few, test_dl_all, device=None, normalize=False):
    data_x, data_y = run_emb_net(emb_net, test_dl_all, device, normalize)
    knn = train_knn(emb_net, test_dl_few, device)

    total_acc = 0
    total_cnt = 0
    for feat, label in zip(data_x, data_y):
        pred = knn.predict(feat[None]).squeeze(0)
        total_acc += pred == label
        total_cnt += 1

    acc = total_acc / total_cnt
    print(f"Accuracy = {acc:.2%} ({total_acc} / {total_cnt})")

    return acc, total_acc, total_cnt


def get_embeddings(model, dataloader, device=None):
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
