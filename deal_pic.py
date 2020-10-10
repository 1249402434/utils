import numpy as np
import cv2
import matplotlib.pyplot as plt


def gaussian_noise(path=None, img=None):
    if not path:
        img = np.array(img)
    else:
        img = cv2.imread(path)
        img = np.array(img)

    if not (img > 1).any():
        # 如果Img的经过归一化到[0,1]了，那么使用如下的方式加入高斯噪声
        noise_img = img + np.random.randn(*img.shape) * np.std(img) / 255
    else:
        noise_img = img + np.random.randn(*img.shape) * np.std(img)

    return noise_img


def salt_and_pepper_noise(path=None, img=None, proportion = 0.05):
    """
    椒盐噪声：就是在图像中随机加一些白点或黑点
    :param path:
    :param img: (h, w, channel)
    :return:
    """
    if not path:
        img = np.array(img)
    else:
        img = cv2.imread(path)
        img = np.array(img)

    height, width, channel = img.shape
    num = int(height * width * proportion)
    for i in range(num):
        h = np.random.randint(0, height)
        w = np.random.randint(0, width)

        if np.random.randint(0, 2) == 0:
            img[h, w, :] = 0
        else:
            img[h, w, :] = 255

    return img


def resize_pic(img):
    pass




image = cv2.imread('./nnd.jpg')
image = salt_and_pepper_noise(img=image)
plt.imshow(image)
plt.axis('off')
plt.show()