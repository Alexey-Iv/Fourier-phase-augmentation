import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset


# hyperparameters
PATH_DATASET = "/kaggle/input/cub2002011/CUB_200_2011/images"
num_classes = 200


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


class Filtered_Dataset(Dataset):
    def __init__(
        self,
        root,
        num_classes=60,
        transform=None,
        split='train',
        seed=42,
        test_size=0.2,
        selected_classes=None,
        samples_split=None
    ):
        self.root = root
        self.num_classes = num_classes
        self.transform = transform
        self.split = split
        self.seed = seed
        self.test_size = test_size

        # Загружаем полный датасет
        self.full_dataset = ImageFolder(root=self.root, transform=None)

        # Определяем классы и разделение
        if selected_classes is None:
            # Случайный выбор классов
            all_classes = os.listdir(self.root)
            self.selected_classes = random.sample(all_classes, self.num_classes)
            self.class_to_idx = {cls: idx for idx, cls in enumerate(self.selected_classes)}

            # Сбор всех образцов для выбранных классов
            all_samples = []
            for path, _ in self.full_dataset.samples:
                class_name = os.path.basename(os.path.dirname(path))
                if class_name in self.selected_classes:
                    all_samples.append((path, self.class_to_idx[class_name]))

            # Разделение на train/val
            train_samples, val_samples = train_test_split(
                all_samples,
                test_size=self.test_size,
                random_state=self.seed,
                stratify=[s[1] for s in all_samples]
            )
            self.samples_split = {'train': train_samples, 'val': val_samples}
        else:
            # Используем переданные классы и разделение
            self.selected_classes = selected_classes
            self.class_to_idx = {cls: idx for idx, cls in enumerate(selected_classes)}
            self.samples_split = samples_split

        # Выбираем нужные данные (train/val)
        self.samples = self.samples_split[split]

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.samples)


# Преобразования
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.4),
    transforms.RandomVerticalFlip(p=0.4),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomRotation(25),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


