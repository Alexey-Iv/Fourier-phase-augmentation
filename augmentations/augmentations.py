import torch
import os
from PIL import Image
from torchvision import transforms
import random
import numpy as np


def augm_batch_of_images(batch, alpha=0.05):
    batch_size, channels, H, W = batch.shape

    h_freq = torch.fft.fftshift(torch.fft.fft2(batch, norm='ortho'))
    amplitude = torch.abs(h_freq)

    phase_origin = torch.angle(h_freq)

    h_ran = h_freq[random.randint(0, batch_size-1)] # [channels, H, W]
    phase_random = torch.angle(h_ran)
    phase_new = alpha * phase_random.unsqueeze(0) + (1 - alpha) * phase_origin
    h_freq_new = amplitude * torch.exp(1j * phase_new)

    output = torch.fft.ifft2(torch.fft.ifftshift(h_freq_new), norm='ortho').real

    return output


def phase_augm(
    input_dir=None,
    output_dir=None,
    photo_length=1000,
    photo_width=1000,
    num_phot=5,
    alpha = 0.05
):
    if not os.path.exists(input_dir):
        raise RuntimeError(f"ERROR, (input_directory) {input_dir} doesn't exist!")

    if num_phot <= 0:
        raise f"num_phot must be bigger than 0!"

    if alpha > 1 or alpha < 0:
        raise f"alpha must be in [0, 1]"


    to_tensor = transforms.ToTensor()
    to_resize = transforms.Resize((photo_length, photo_width))

    os.makedirs(output_dir, exist_ok=True)

    count_files = len(os.listdir(input_dir))
    count_batches = count_files // num_phot + 1

    begin_index = 0
    for i in range(count_batches):
        batch_files = os.listdir(input_dir)[begin_index:begin_index+num_phot]

        tensors = []
        for filename in batch_files:
            try:
                img_path = os.path.join(input_dir, filename)
                img = Image.open(img_path).convert("RGB")

                tensor = to_tensor(to_resize(img))
                tensors.append(tensor)
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
                continue

        if tensors:
            batch_tensor = torch.stack(tensors)

            augmented_batch = augm_batch_of_images(batch_tensor, alpha)

            for idx, augmented_tensor in enumerate(augmented_batch):
                arr = augmented_tensor.permute(1, 2, 0).numpy()  # CHW → HWC
                arr = (arr * 255).clip(0, 255).astype(np.uint8)  # Масштабирование
                im = Image.fromarray(arr)

                base_name = os.path.splitext(batch_files[idx])[0]
                output_path = os.path.join(output_dir, f"{base_name}_aug.png")

                im.save(output_path)

        begin_index += num_phot

# TODO
# сделать функцию, для добавления такой аугментации к подпапкам в изначальной директории.
# чтобы все фотографии могли быть корректно обработаны.

if __name__ == "__main__":
    # example of using
    phase_augm("./Small/002/R", "./evrvr", 62, 512, 2)
