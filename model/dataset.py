from torchvision.transforms import transforms
import torch
from collections import defaultdict
import glob
import os
from PIL import Image
import random


#(64, 512, 3)
norm_transform = transforms.Compose([
    transforms.Resize((224, 224)),
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

        self.need_classes = []
        self.class_to_idxs = defaultdict(list)

        all_patients = sorted(
            glob.glob(os.path.join(self.root, '*')),
            key=lambda x: int(os.path.basename(x))
        )

        self.data = []
        if mode == "train":
            self.patients = all_patients[:num_seen_classes]
            self.need_classes = list(range(num_seen_classes))

            for patient_dir in self.patients:
                patient_id = int(os.path.basename(patient_dir))
                for eye_dir in ['L', 'R']:
                    eye_path = os.path.join(patient_dir, eye_dir)
                    if os.path.exists(eye_path):
                        images = glob.glob(os.path.join(eye_path, '*.*'))
                        self.data.extend([(img, patient_id) for img in images])

        elif mode in ["test_few", "test_all"]:
            self.patients = all_patients[num_seen_classes:]
            self.need_classes = [i for i in range(len(all_patients) - num_seen_classes)]

            for patient_dir in self.patients:
                patient_id = int(os.path.basename(patient_dir))
                for eye_dir in ['L', 'R']:
                    eye_path = os.path.join(patient_dir, eye_dir)
                    if os.path.exists(eye_path):
                        images = glob.glob(os.path.join(eye_path, '*.*'))
                        if mode == "test_few":
                            selected_images = images[:1]
                        else:
                            selected_images = images[1:]
                        self.data.extend([(img, patient_id) for img in selected_images])
        else:
            raise ValueError("Invalid mode. Use 'train' or 'test'")

        for idx, (_, label) in enumerate(self.data):
            self.class_to_idxs[label-1].append(idx)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_path, label = self.data[index]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label-1


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

