import numpy as np
import matplotlib.pyplot as plt
import imageio
from PIL import Image
from io import BytesIO
import os


def create_animation_from_dataset(frames_count=10, test_dataset=None, output_path="animation.gif"):
    frames = []

    for i in range(frames_count):
        image, label = test_dataset[i]
        out = model(image.unsqueeze(0).cuda())
        predict = torch.max(out.data, 1)[1][0]

        img = image.permute(1, 2, 0)

        fig = plt.figure(figsize=(6, 6))
        plt.text(
            x=0.05,
            y=0.95,
            s=f"True: {label}        Pred: {predict}",
            color="Red",
            fontsize=14,
            ha="left",
            va="top",
            transform=plt.gca().transAxes
        )

        plt.imshow(img, cmap='gray')
        plt.axis('off')

        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        buf.seek(0)
        frames.append(Image.open(buf))
        plt.close()

    # Saving the gif
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=500,
        loop=0
    )

    print(f"GIF сохранен в {output_path}")


def create_animation_from_dir(root_dir=None, output_path="animation.gif"):
    frames = []

    for i in is.listdir(root_dir):
        img = Image.open(file).convert("RGB")

        fig = plt.figure(figsize=(6, 6))
        plt.imshow(img)
        plt.axis('off')

        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        buf.seek(0)
        frames.append(Image.open(buf))
        plt.close()

    # Сохраняем как GIF
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=1500,
        loop=0
    )

    print(f"GIF сохранен в {output_path}")



if __name__ == "__main__":
    # Example
    test_dataset = ["your dataset"]
    create_animation(20, test_dataset, "iris_animation.gif")
