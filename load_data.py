import numpy as np
from skimage.io import imread
from skimage.transform import resize
import pandas as pd
import os



def load_img(imgpath):
    N = 33431
    print('Number of images:', N)
    name = [str(i)+'.jpg' for i in range(N)]
    img = np.zeros([N, 64, 64, 3])
    for n, file in enumerate(name):
        img[n] = resize(imread(os.path.join(imgpath, file)), (64,64,3))
    return img

def text2vector(textpath, vdim, test=False):
    ### check vdim > 23
    ### Tokenizer vocabulary
    colorhair = ['orange hair', 'white hair', 'aqua hair', 'gray hair','green hair', 'red hair',
                 'purple hair', 'pink hair','blue hair', 'black hair', 'brown hair', 'blonde hair']
    coloreyes = ['gray eyes', 'black eyes', 'orange eyes','pink eyes', 'yellow eyes',
                 'aqua eyes', 'purple eyes','green eyes', 'brown eyes', 'red eyes', 'blue eyes']
    tags = colorhair + coloreyes

    ### text2vector
    readtags = pd.read_csv(textpath, sep=',', header=None).values
    L = readtags.shape[0]
    text_vector = np.zeros([L, vdim])
    for i in range(L):
        tag_sentence = readtags[i,1]
        for j, tag in enumerate(tags):
            if tag in tag_sentence:
                text_vector[i,j] = 1.
    if test:
        return text_vector, readtags[:,0]
    else:
        return text_vector


if __name__ == '__main__':
    text_vector = text2vector('tags_clean.csv', 50)
    