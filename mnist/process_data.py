# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 23:01:05 2019

@author: jingh
"""

import os
import struct
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

def load_mnist(path, kind='train'):
    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte'
                               % kind)
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II',
                                 lbpath.read(8))
        labels = np.fromfile(lbpath,
                             dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII',
                                               imgpath.read(16))
        images = np.fromfile(imgpath,
                             dtype=np.uint8).reshape(len(labels), 784)

    return images, labels


path = '.\mnist'
images, labels = load_mnist(path, 'train')
images_test, labels_test = load_mnist(path, 't10k')


# see the 12th image
im = images[12,:]
im = im.reshape(28, 28) 

# show the image
plt.figure()
plt.imshow(im, cmap='gray')

path = './train'
if not os.path.exists(path):
    os.makedirs(path)
for i in range(images.shape[0]):
    im = images[i, :]
    im1 = im.reshape(28, 28)
    im2 = Image.fromarray(im1)
    name = os.path.join(path, str(i)+'.png')
    im2.save(name)

path = './test'
if not os.path.exists(path):
    os.makedirs(path)
for i in range(images_test.shape[0]):
    im = images_test[i, :]
    im1 = im.reshape(28, 28)
    im2 = Image.fromarray(im1)
    name = os.path.join(path, str(i)+'.png')
    im2.save(name)

# draw all figures
fig, ax = plt.subplots(
    nrows=2,
    ncols=5,
    sharex=True,
    sharey=True, )

ax = ax.flatten()
for i in range(10):
    img = images[labels == i][0].reshape(28, 28)
    ax[i].imshow(img, cmap='Greys', interpolation='nearest')

ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()


# show some figures of label 7
fig, ax = plt.subplots(
    nrows=5,
    ncols=5,
    sharex=True,
    sharey=True, )

ax = ax.flatten()
for i in range(25):
    img = images[labels == 7][i].reshape(28, 28)
    ax[i].imshow(img, cmap='Greys', interpolation='nearest')

ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.show()

plt.show()



plt.show()
