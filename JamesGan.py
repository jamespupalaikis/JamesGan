import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow.keras as k
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LeakyReLU, Conv2D, MaxPool2D, Flatten, Dropout, Reshape, Conv2DTranspose
from tensorflow.keras.optimizers import Adam
from keras.models import load_model
import os
import random
from PIL import Image
from keras import backend


def wasserstein_loss(y_true, y_pred):
    return backend.mean(y_true * y_pred)


learnd = 0.0001
learng = 0.0001


def define_discriminator(in_shape=(128, 128, 3)):
    model = Sequential([
        Conv2D(128, (3, 3), (2, 2), padding='same', input_shape=in_shape),
        LeakyReLU(alpha=0.2),
        Conv2D(128, (3, 3), (2, 2)),
        LeakyReLU(alpha=0.2),
        Conv2D(128, (3, 3), (2, 2)),
        LeakyReLU(alpha=0.2),
        Flatten(),
        Dropout(.5),
        Dense(64),  # * 16*16),
        Dense(1, activation='sigmoid')

    ])

    model.compile(loss='binary_crossentropy',
                  optimizer=Adam(lr=learnd, beta_1=0.5),  # optimizer=Adam(lr = 0.0002,beta_1=0.5),
                  metrics=['accuracy'])
    return model


def define_generator(latent_dim):
    n_nodes = 128 * 16 * 16  # for creating 128 copies of 16x16 res output images for upsampling ######

    #####x3?############
    model = Sequential([
        Dense(64, input_dim=latent_dim),#I added another densely connected layer here,
        Dropout(0.3),                   #against standard practice, because I found it allowed for a
        Dense(n_nodes),                 #better plateau on image quality
        LeakyReLU(alpha=0.2),
        Reshape((16, 16, 128)),
        # upsample to 32x32
        Conv2DTranspose(128, (4, 4), (2, 2), padding='same'),
        LeakyReLU(alpha=0.2),
        #now to 64x64
        Conv2DTranspose(128, (4, 4), (2, 2), padding='same'),
        LeakyReLU(alpha=0.2),
        #and finally 128x128
        Conv2DTranspose(128, (4, 4), (2, 2), padding='same'),
        LeakyReLU(alpha=0.2),
        # generate
        Conv2D(3, (16, 16), activation='tanh', padding='same')
    ])
    return model


def define_gan(generator, discriminator):
    discriminator.trainable = False #only train the generator here
    model = Sequential([
        generator,
        discriminator
    ])
    model.compile(loss='binary_crossentropy', optimizer=
    Adam(lr=learng, beta_1=0.5))
    return model


def load_real_samples(url):
    print('loading images...')
    percent = .90 #defines the percent of images that will be used (chosen randomly)
    dir = url
    dir1 = os.listdir(dir)
    full = int(len(dir1) * percent) - 1
    imagelist = []
    files = random.sample(dir1, full)
    j = 1

    for i in files:
        print(j, full)
        j += 1
        path = os.path.join(dir, i)
        im = Image.open(path)
        pix = list(im.getdata())
        width, height = im.size
        pixels = [pix[i * width:(i + 1) * width] for i in range(height)]
        for row in pixels:
            for colnum in range(len(row)):
                (x, y, z) = row[colnum]
                row[colnum] = [x, y, z]
        pixels = np.array(pixels)
        imagelist.append(pixels)
    imagelist = np.array(imagelist)
    imagelist = imagelist.astype('float32')
    imagelist = (imagelist - 127.5) / 127.5#normalize images
    return imagelist


def gen_real_samples(dataset, n_samples):  #samples random real examples for discriminator training
    ix = np.random.randint(0, dataset.shape[0], n_samples)

    x = dataset[ix]
    # y = np.randn
    y1 = np.random.random([n_samples, 1]) * 0.105  #label smoothing for better convergence
    y = np.ones((n_samples, 1))
    y = y - y1
    return x, y


def gen_latent_points(latent_dim, n_samples): #generate latent noise for the generator
    xinput = np.random.randn(latent_dim * n_samples)
    xinput = xinput.reshape(n_samples, latent_dim)
    return xinput


def gen_fake_samples(generator, latent_dim, n_samples): #generate fake samples for discriminator training
    xinput = gen_latent_points(latent_dim, n_samples)
    x = generator.predict(xinput)
    y1 = np.random.random([n_samples, 1]) * 0.105 #label smoothing
    y = np.zeros((n_samples, 1))
    y = y + y1
    return x, y


def train(gen, disc, gan, dataset, latent_dim, n_epochs, n_batch, save=True): #training protocol
    bat_per_epo = int(dataset.shape[0] / n_batch)
    halfbatch = int(n_batch / 2)
    print('starting up...')
    for i in range(n_epochs):
        # enum batches over training set
        for j in range(bat_per_epo):
            xreal, yreal = gen_real_samples(dataset, halfbatch)
            xfake, yfake = gen_fake_samples(gen, latent_dim, halfbatch)
            disc_loss_1, _ = disc.train_on_batch(xreal, yreal)
            disc_loss_2, _ = disc.train_on_batch(xfake, yfake)

            xgan = gen_latent_points(latent_dim, n_batch)
            ygan = np.ones((n_batch, 1))

            gan_loss = gan.train_on_batch(xgan, ygan)

            print('>%d,%d/%d, d1 = %.3f, d2 = %.3f g = %.3f' %
                  (i + 1, j + 1, bat_per_epo, disc_loss_1, disc_loss_2, gan_loss))
    if (save == True):
        gen.save('models/jamesgan%d_ld_%d_ep_gen.h5' % (latent_dim, n_epochs))
        disc.save('models/jamesgan%d_ld_%d_ep_discrim.h5' % (latent_dim, n_epochs))



def interval_train_no_load(gen, disc, gan, dataset, latent_dim, n_epochs, n_batch,runs,start = 0 ):
    '''this function will train the model for [n_epochs] * [runs] epochs in total,
        saving a new model every [n_epochs]. It will name the models according to the number of
        epochs passed, and will begin from [start] epochs in the filenames.'''
    eps = start
    for i in range (runs):
        print('beginning run number %d, starting with epoch %d'%(i, eps))
        eps += n_epochs
        train(gen, disc, gan, dataset, latent_dim, n_epochs, n_batch, False)
        gen.save('models/jamesgan%d_ld_%d_ep_gen.h5'%(latent_dim, eps))
        disc.save('models/jamesgan%d_ld_%d_ep_discrim.h5'%(latent_dim, eps))




###################################END CELL 1 IF IN IPYNB##################


latent_dim = 100#latent dimensions
discriminator = define_discriminator()
generator = define_generator(latent_dim)
gan = define_gan(generator, discriminator)
epochs = 10#number of epochs per update
n_batch = 128#batch size
runs = 100


################END CELL 2############################

dataset = load_real_samples('Data')#load samples from 'data' folder (this can take a while)
interval_train_no_load(generator,discriminator,gan,dataset,latent_dim,epochs,n_batch,runs)#train for 1000 epochs, saving every 10


