import numpy as np
from matplotlib import pyplot as plt

def imgflatten(img):
    X,y = [],[]

    imgrange = img.max() - img.min()
    img = (img - img.min())/imgrange
    
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            X.append([i,j])
            if type(img[i,j]) is np.ndarray:
                y.append(img[i,j])
            else:
                y.append([img[i,j]])

    return np.array(X), np.array(y), img.shape


def imgdeflatten(X,y,shape):
    img = np.zeros(shape)
    for (i,j),val in zip(X,y):
        img[i,j] = val

    # imgrange = img.max() - img.min()
    # img = (img - img.min())/imgrange

    return img
