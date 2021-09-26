import matplotlib.pyplot as plt

from PIL import Image


def show_image():
    image1 = Image.open(f"./Training400/train/Glaucoma/g0001.jpg")
    image2 = Image.open(f"./Training400/train/Glaucoma/g0002.jpg")
    image3 = Image.open(f"./Training400/train/Glaucoma/g0003.jpg")
    image4 = Image.open(f"./Training400/train/Non-Glaucoma/n0001.jpg")
    image5 = Image.open(f"./Training400/train/Non-Glaucoma/n0002.jpg")
    image6 = Image.open(f"./Training400/train/Non-Glaucoma/n0003.jpg")

    plt.figure(figsize=(20, 10))  # 设置窗口大小
    plt.suptitle('Glaucoma And Non-Glaucoma')
    plt.subplot(2, 3, 1)
    plt.title('Glaucoma')
    plt.imshow(image1)
    plt.axis('off')

    plt.subplot(2, 3, 2)
    plt.title('Glaucoma')
    plt.imshow(image2)
    plt.axis('off')

    plt.subplot(2, 3, 3)
    plt.title('Glaucoma')
    plt.imshow(image3)
    plt.axis('off')

    plt.subplot(2, 3, 4)
    plt.title('Non-Glaucoma')
    plt.imshow(image4)
    plt.axis('off')

    plt.subplot(2, 3, 5)
    plt.title('Non-Glaucoma')
    plt.imshow(image5)
    plt.axis('off')

    plt.subplot(2, 3, 6)
    plt.title('Non-Glaucoma')
    plt.imshow(image6)
    plt.axis('off')

    plt.show()


if __name__ == '__main__':
    show_image()
