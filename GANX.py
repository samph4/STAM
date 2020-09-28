# Global Variables 
from numpy import hstack
from numpy import zeros
from numpy import ones
from numpy.random import rand
from numpy.random import randn
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras.layers import LeakyReLU
from keras.layers import BatchNormalization
from keras import optimizers
from keras import initializers
from matplotlib import pyplot
from pylab import rcParams
import os
import math
import pandas as pd
import numpy as np
import warnings
import random
from scipy.optimize import curve_fit
from tensorflow.python.client import device_lib
import tensorflow as tf
warnings.filterwarnings(action='once')
path = os.getcwd()

#Global Variables
a = 1
p = 3
blv = 100
max_velocity = 600
n = 601
path = os.getcwd()

neval = 1000
impact_range = np.linspace(0,max_velocity,n)
foldername = '/Completed Simulations/Maisie/2/'

# Functions relating to the GAN network
def generate_real_samples(n, training_set):
    #n=600
    np.random.seed(30)
    idx = np.random.randint(len(training_set), size=int(n))
    #idx = np.random.randint(len(dataset.values), size=int(n))
    X = training_set[idx,:]
    #X = dataset.values[idx,:]
    y = ones((n, 1))
    return X, y

# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n):
    # generate points in the latent space
    x_input = randn(latent_dim * n)
    # reshape into a batch of inputs for the network
    x_input = x_input.reshape(n, latent_dim)
    return x_input

    # use the generator to generate n fake examples, with class labels

#use generator to generate fake samples
def generate_fake_samples(generator, latent_dim, n):
    # generate points in latent space
    x_input = generate_latent_points(latent_dim, n)
    # predict outputs
    X = generator.predict(x_input)
    # create class labels
    y = zeros((n, 1))
    return X, y

# lambert model for analysis
def lambert_model(impact_velocity, a, p, blv):
    
    residual_velocity = []
    for impact_v in impact_velocity:
        
        if impact_v**2 - blv**2 < 0: #if loop to eliminate negative sqrt values at impact velocities beneath the BLV
            vr = 0
        else:
            vr = a*((impact_v**p) - (blv**p))**(1/p)   
        residual_velocity.append(vr)      
    
    return residual_velocity

# evaluate the discriminator and plot real and fake points
def summarize_performance(epoch, generator, discriminator, training_set, latent_dim, lr, i, RMSE_train, a_x,p_x,blv_x,D_losses, n=150):
    
    '''

    Update trainable parameters of the model, also prints output to command line indicating iteration

    '''
    # prepare real samples
    x_real, y_real = generate_real_samples(n, training_set)
    # evaluate discriminator on real examples
    _, acc_real = discriminator.evaluate(x_real, y_real, verbose=0)
    # prepare fake examples
    x_fake, y_fake = generate_fake_samples(generator, latent_dim, n)
    # evaluate discriminator on fake examples
    _, acc_fake = discriminator.evaluate(x_fake, y_fake, verbose=0)
    # summarize discriminator performance
    print('Iteration #: ', (epoch+1))

    '''

    Construct log information/data; includes RMSE, percentage difference and curve fitting

    '''
    #RMSE calculations
    x,y = generate_fake_samples(generator, latent_dim, 1000)
    x1 = x # modified generated samples

    for i in range(0,len(x)): #first constraint
        if x1[i,1] < 0:
            x1[i,1] = 0

    x2 = x1
    columnIndex = 0
    x2 = x2[x2[:,columnIndex].argsort()]
    
    # Log Performance
    losses = [acc_real, acc_fake]; lambert_rv = lambert_model(x2[:,0], 1, 3, 100); difference = (sum((x2[:,1] - lambert_rv)**2)/len(lambert_rv))**0.5
    RMSE_train.append(difference); D_losses.append(losses)
    
    # ___________CURVE FITTING________________
    fitted = []
    try:
        param, param_cov = curve_fit(lambert_model, x_fake[:,0], x_fake[:,1], maxfev=10000) 
         # fitted ydata
        fitted_x = np.sort(x_fake[:,0])   
        for impact_v in fitted_x:
            if impact_v**2 - param[2]**2 < 0:
                vr = 0
            else:
                vr = param[0]*((impact_v**param[1]) - (param[2]**param[1]))**(1/param[1]) 
            fitted.append(vr) 
            
        # % difference
        xa=(abs(param[0]-a)/((param[0]+a)/2))*100
        a_x.append(xa)
        xp=(abs(param[1]-p)/((param[1]+p)/2))*100
        p_x.append(xp)
        xblv=(abs(param[2]-blv)/((param[2]+blv)/2))*100
        blv_x.append(xblv)

    except RuntimeError:
        fitted.append(np.nan)
        a_x.append(np.nan)
        p_x.append(np.nan)
        blv_x.append(np.nan)
        fitted_x = np.sort(x_fake[:,0])
        param, param_cov = np.nan, np.nan

    print('  RMSE =    ', difference, '%')

    generator.save(path + foldername + "LR_%s/Networks/generator_%d.h5"%(lr, epoch+1))
    discriminator.save(path + foldername + "LR_%s/Networks/model_%d.h5"%(lr, epoch+1)) 

    if (i+1) % 1000 == 0:
        # _______ GENERATED PLOT___________
        pyplot.subplot(1,2,1)
        pyplot.scatter(training_set[:,0], training_set[:,1], marker = 'x', c = 'r', label='Scatter')
        #pyplot.plot(impact_range, residual_velocity, 'k--', label='Lambert | a = %0.2f p = %0.2f blv = %0.2f' % (a,p,blv))
        #pyplot.plot(fitted_x, fitted,'--',color='navy', label='fitted     | a = %0.2f p = %0.2f blv = %0.2f' % (param[0],param[1],param[2]))
        pyplot.scatter(x_fake[:, 0], x_fake[:, 1], color='navy', s = 20, alpha = 0.4,label='GAN Output')
        #yx = pyplot.plot(range(600), range(600),'k',alpha=0.2)
        
        pyplot.grid(True)
        pyplot.xlabel('Impact Velocity [m/s]')
        pyplot.ylabel('Residual Velocity [m/s]')
        pyplot.title('                                                                       Lr = %s @ %d' % (lr,(epoch+1)))
        pyplot.legend(framealpha=1, borderpad=1, edgecolor = 'k',loc=2)
        #pyplot.ylim(-200,700)
        #pyplot.xlim(0,700)
        pyplot.grid(b=None) 
        pyplot.xlim(0,900)
        pyplot.ylim(0,900)
        # _________RMSE PLOT___________
        pyplot.subplot(1,2,2)
        if len(RMSE_train) == 1:
            pyplot.plot(1000,RMSE_train)
            pyplot.xlabel('Number of Iterations')
            pyplot.ylabel('RMSE [%]')
            pyplot.title('                                                                            % Difference')
            pyplot.grid(b=None)
        else:        
            pyplot.plot(np.linspace(0,len(RMSE_train)*neval,len(RMSE_train)), a_x, label = 'a')
            pyplot.plot(np.linspace(0,len(RMSE_train)*neval,len(RMSE_train)), p_x, label = 'p')
            pyplot.plot(np.linspace(0,len(RMSE_train)*neval,len(RMSE_train)), blv_x, label = 'blv')
            pyplot.plot(np.linspace(0,len(RMSE_train)*neval,len(RMSE_train)), RMSE_train, 'k', label = 'RMSE')
            pyplot.xlabel('Number of Iterations')
            pyplot.ylabel('RMSE [%]')
            pyplot.title('                                                                            % Difference')
            pyplot.grid(b=None)
        print('  RMSE =    ', difference, '%')

        pyplot.legend(framealpha=1, borderpad=1, edgecolor = 'k',loc=1)
        pyplot.ylim(0,300)
        pyplot.savefig(path + foldername + "LR_%s/Plots/Iteration_%d"%(lr, epoch+1))
        generator.save(path + foldername + "LR_%s/Networks/generator_%d.h5"%(lr, epoch+1))
        discriminator.save(path + foldername + "LR_%s/Networks/model_%d.h5"%(lr, epoch+1)) 
        pyplot.show()

#train the generator and discriminator
def train(g_model, d_model, gan_model, training_set, latent_dim, lr, RMSE_train, a_x,p_x,blv_x,D_losses, n_epochs=10000, n_batch=128, n_eval=2000):
    # determine half the size of one batch, for updating the discriminator
    half_batch = int(n_batch / 2)
    # manually enumerate epochs
    for i in range(n_epochs):
        # prepare real samples
        x_real, y_real = generate_real_samples(half_batch, training_set)
        # prepare fake examples
        x_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
        # update discriminator
        d_model.train_on_batch(x_real, y_real)
        d_model.train_on_batch(x_fake, y_fake)
        # prepare points in latent space as input for the generator
        x_gan = generate_latent_points(latent_dim, n_batch)
        # create inverted labels for the fake samples
        y_gan = ones((n_batch, 1))
        # update the generator via the discriminator's error
        gan_model.train_on_batch(x_gan, y_gan)
        # evaluate the model every n_eval epochs
        if (i+1) % n_eval == 0:
            summarize_performance(i, g_model, d_model, training_set,latent_dim, lr, i, RMSE_train, a_x,p_x,blv_x,D_losses,)
            



