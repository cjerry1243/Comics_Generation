import numpy as np
from skimage.io import imsave, imshow

from keras.models import Sequential, Model
from keras.layers import Flatten, Input, Concatenate, Conv2DTranspose, Conv2D, UpSampling2D
from keras.layers import BatchNormalization, Activation, Dropout, Dense, Reshape
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam, RMSprop
from keras import backend as K

from load_data import load_img, text2vector

def make_trainable(net, val):
    net.trainable = val
    for l in net.layers:
        l.trainable = val

def GAN_0():
    dropout_rate = 0.3
    # Build Generative model ...
    nch = 128
    g_input = Input(shape=[vdim + zdim])
    H = Dense(nch * 32 * 32, init='glorot_normal')(g_input)
    H = BatchNormalization()(H)
    H = Activation('relu')(H)
    H = Reshape([32, 32, nch])(H)
    H = UpSampling2D(size=(2, 2))(H)
    H = Conv2D(64, [3, 3], padding='same')(H)
    H = BatchNormalization()(H)
    H = Activation('relu')(H)
    H = Conv2D(32, [3, 3], padding='same')(H)
    H = BatchNormalization()(H)
    H = Activation('relu')(H)
    H = Conv2D(3, [1, 1], padding='same')(H)
    g_V = Activation('sigmoid')(H)
    generator = Model(g_input, g_V)
    generator.summary()
    # Build Discriminative model ...
    d_input = Input(shape=(64, 64, 3))
    H = Conv2D(128, [5, 5], strides=(2, 2), padding='same')(d_input)
    H = LeakyReLU(0.2)(H)
    H = Dropout(dropout_rate)(H)
    H = Conv2D(256, [5, 5], strides=(2, 2), padding='same')(H)
    H = LeakyReLU(0.2)(H)
    H = Dropout(dropout_rate)(H)
    H = Conv2D(512, [5, 5], strides=(2, 2), padding='same')(H)
    H = LeakyReLU(0.2)(H)
    H = Dropout(dropout_rate)(H)
    H = Flatten()(H)
    v_input = Input(shape=[vdim])
    H = Concatenate(axis=-1)([H, v_input])
    H = Dense(256)(H)
    H = LeakyReLU(0.2)(H)
    H = Dropout(dropout_rate)(H)
    d_V = Dense(2, activation='softmax')(H)
    discriminator = Model([d_input, v_input],d_V)
    discriminator.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['acc'])
    discriminator.summary()
    # Freeze weights in the discriminator for stacked training

    make_trainable(discriminator, False)
    # Build stacked GAN model
    gan_input = Input(shape=[vdim + zdim])
    H = generator(gan_input)
    v_input = Input(shape=[vdim])
    gan_V = discriminator([H, v_input])
    GAN = Model([gan_input, v_input], gan_V)
    GAN.compile(optimizer=adam1, loss='categorical_crossentropy', metrics=['acc'])
    GAN.summary()
    return generator, discriminator, GAN

def wrongtext(text, tags):
    wrong = np.zeros_like(text)
    for i in range(text.shape[0]):
        keep = True
        while keep:
            num = np.random.randint(tags.shape[0])
            if np.sum(text[i]*tags[num]) < 2:
                keep = False
                wrong[i] = tags[num]
    return wrong

units = 64
maxepoch = 500000
vdim = 30
zdim = 100
batchsize = 64
adam = Adam(lr=0.0002)
adam1 = Adam(lr=0.0002)

if __name__ == '__main__':
    G, D, GAN = GAN_0()
    # train_D, train_G = WGAN_training()
    print('Load data...')
    right_text = text2vector('tags_clean.csv', vdim=vdim)
    tags = np.unique(right_text, axis=0)[1:]
    tags = tags[tags.sum(axis=1) == 2]  # shape = (190, vdim)

    # real_image = load_img('faces')
    # np.save('faces.npy', real_image)
    # exit()

    real_image = np.load('faces01.npy')
    # print(real_image[0])

    N = real_image.shape[0]
    # iteration = N//batchsize

    D_train_loss, G_train_loss = [], []
    for epoch in range(maxepoch):
        shuffle = np.random.permutation(N)
        batch = shuffle[:batchsize]
        ### Real image
        I = real_image[batch]

        ### Reshape vector into shape=(N,1,1,vdim)
        right_V = right_text[batch]# .reshape(batchsize, 1, 1, vdim)
        wrong_V = wrongtext(right_V, tags)# .reshape(batchsize, 1, 1, vdim)

        ### concatenate v, z
        z = np.random.uniform(0,1,size=(batchsize, zdim))
        v_z = np.concatenate((right_V, z), axis=-1)
        fake_I = G.predict(v_z)

        ### Update D
        img_input = np.concatenate((I, I, fake_I), axis=0)
        text_input = np.concatenate((right_V, wrong_V, right_V), axis=0)
        # loss_D = train_D([img_input, text_input])
        D_y = np.zeros([batchsize*3, 2])
        D_y[0:batchsize, 1] = 1.
        D_y[batchsize:, 0] = 1.
        loss_D , acc_D = D.train_on_batch([img_input, text_input], D_y)
        D_train_loss.append(loss_D)

        ### Update G
        GAN_y = D_y[0:batchsize]
        loss_G, acc_G = GAN.train_on_batch([v_z, right_V], GAN_y)
        G_train_loss.append(loss_G)

        ### Save Weights and Print loss
        print('epoch:', epoch, 'loss_D:', loss_D, 'acc_D:', acc_D, 'loss_G:',loss_G, 'acc_G:', acc_G,)

        if (epoch+1)%10 ==0:
            G.save_weights('GAN_G_weights.h5')
            D.save_weights('GAN_D_weights.h5')
            np.save('GAN_Dloss.npy', loss_D)
            np.save('GAN_Gloss.npy', loss_G)

        ### Save 2 Images
        if (epoch + 1) % 500 == 0:
            for s in range(5):
                imsave('img/' + str(epoch) + '_' + str(shuffle[s]) + '.jpg', fake_I[s])
                # imshow(fake_image[s])