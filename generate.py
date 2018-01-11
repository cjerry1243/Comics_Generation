import numpy as np
from load_data import text2vector
from train import GAN_0, vdim, zdim
from skimage.io import imsave
import sys

G, D, G_D = GAN_0()
G.load_weights('GAN_G_weights0003.h5')
path = sys.argv[1]
text, index = text2vector(path, vdim=vdim, test=True)
images = np.zeros([64*text.shape[0], 64*5, 3])

for i in range(text.shape[0]):
    v = text[i:(i+1)]
    for j in range(5):
        np.random.seed(1000+j)
        z = np.random.uniform(0.3, 0.7, size=(1, zdim))
        v_z = np.concatenate((v, z), axis=-1)
        G_image = G.predict(v_z)
        images[64*i:64*(i+1),64*j:64*(j+1),:] = G_image
        ### Save images: 'Samples/sample_1_1.jpg'
        imsave('samples/sample_'+str(index[i])+'_'+str(j+1)+'.jpg',
               G_image[0])

import matplotlib.pyplot as plt
plt.imshow(images)
plt.axis('off')
plt.show()