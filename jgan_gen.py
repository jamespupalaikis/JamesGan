from keras.models import load_model
from numpy.random import randn
from matplotlib import pyplot
import pandas as pd
import numpy as np

'''this file contains functions that will generate images from a trained model'''

# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples):
    # generate points in the latent space
    x_input = randn(latent_dim * n_samples)
    # reshape into a batch of inputs for the network
    x_input = x_input.reshape(n_samples, latent_dim)
    return x_input



# create a plot of generated images
def show_plot(examples, n):
    # plot images
    for i in range(n * n):
        # define subplot
        pyplot.subplot(n, n, 1 + i)
        # turn off axis
        pyplot.axis('off')
        # plot raw pixel data
        pyplot.imshow(examples[i, :, :, 0])#, cmap='gray_r')
    pyplot.show()



def gen_grid(wide): #generate a [wide] by [wide] grid of random samples
    latent_points = generate_latent_points(ld, wide * wide)
    X = model.predict(latent_points)
    np.savetxt('latents.csv', latent_points)
    X = ((X * 125) + 125)
    X = X.astype('int')
    show_plot(X, wide)



def gen_realcolor(): #generate a random image in RGB color
    latent_points = generate_latent_points(ld,1)
    X = model.predict(latent_points)
    np.savetxt('latents.csv', latent_points, delimiter = ',')
    X  = ((X * 125) + 125)
    X =  X.astype('int')
    pyplot.imshow(X[0])
    pyplot.show()

def gen_fromlatent(i):
    '''the previous programs will save the last used latent points as "latents.csv".
    This function will load the ith index of that csv and generate an RGB image from it'''
    latents = np.genfromtxt('latents.csv')
    X = model.predict(latents)
    X = ((X * 125) + 125)
    X = X.astype('int')
    pyplot.imshow(X[i])
    pyplot.show()


epochs =4900#epochs of the desired model to load
ld = 100#latent dimensions of the model

# load model
model = load_model('models/jamesgan%d_ld_%d_ep_gen_.h5' %(ld,epochs))
gen_grid(3)





