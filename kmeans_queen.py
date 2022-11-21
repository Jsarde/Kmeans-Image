from PIL import Image, ImageOps
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


with Image.open('../Data/queen.png') as img:
    gray_image = ImageOps.grayscale(img)
    arr_img = np.array(gray_image)
    flatten_image = np.ndarray.flatten(arr_img,order='C')
    flatten_image = np.expand_dims(flatten_image,axis=1)
    ax = plt.subplot(2, 2, 1)
    ax.imshow(img)
    ax.set_title('Original Queen')
    for k in range(2,4+1,1):
        model = KMeans(n_clusters=k)
        model.fit(flatten_image)
        res = model.labels_
        img2 = np.reshape(res,newshape=arr_img.shape)
        ax = plt.subplot(2,2,k)
        ax.imshow(img2)
        ax.set_title(f'{k} Cluster')
    plt.show()