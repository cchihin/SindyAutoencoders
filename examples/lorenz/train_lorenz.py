#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
sys.path.append("../../src")
import os
import datetime
import pandas as pd
import numpy as np
from example_lorenz import get_lorenz_data
from sindy_utils import library_size
from training import train_network
import tensorflow as tf


# In[2]:


tf.config.list_physical_devices('GPU')


# # Generate data

# In[3]:


# generate training, validation, testing data
noise_strength = 1e-6
training_data = get_lorenz_data(1024, noise_strength=noise_strength)
validation_data = get_lorenz_data(20, noise_strength=noise_strength)


# In[4]:


training_data.keys()


# In[5]:


np.shape(training_data['sindy_coefficients'])


# # Set up model and training parameters

# In[6]:


params = {}

params['input_dim'] = 128      # Input dimension
params['latent_dim'] = 3       # Latent dimension
params['model_order'] = 1      # Order of derivative
params['poly_order'] = 3       # Polynomial order
params['include_sine'] = False # Using sinc functions
params['library_dim'] = library_size(params['latent_dim'], params['poly_order'], params['include_sine'], True)

# sequential thresholding parameters
params['sequential_thresholding'] = True
params['coefficient_threshold'] = 0.1
params['threshold_frequency'] = 500
params['coefficient_mask'] = np.ones((params['library_dim'], params['latent_dim']))
params['coefficient_initialization'] = 'constant'

# loss function weighting
params['loss_weight_decoder'] = 1.0
params['loss_weight_sindy_z'] = 0.0
params['loss_weight_sindy_x'] = 1e-4
params['loss_weight_sindy_regularization'] = 1e-5

params['activation'] = 'sigmoid'
params['widths'] = [64,32]


# training parameters
params['epoch_size'] = training_data['x'].shape[0]
# params['batch_size'] = 1024
params['batch_size'] = 8192

params['learning_rate'] = 1e-3

params['data_path'] = os.getcwd() + '/'
params['print_progress'] = True
params['print_frequency'] = 100

# training time cutoffs
params['max_epochs'] = 10001
params['refinement_epochs'] = 1001


# Run training experiments

num_experiments = 10
df = pd.DataFrame()
for i in range(num_experiments):
    print('EXPERIMENT %d' % i)

    params['coefficient_mask'] = np.ones((params['library_dim'], params['latent_dim']))

    params['save_name'] = 'lorenz_' + datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")

    tf.compat.v1.reset_default_graph()

    results_dict = train_network(training_data, validation_data, params)
    # df = df.append({**results_dict, **params}, ignore_index=True)
    df_newrow = pd.DataFrame([{**results_dict, **params}])
    df = pd.concat([df, df_newrow], ignore_index=True)


df.to_pickle('experiment_results_' + datetime.datetime.now().strftime("%Y%m%d%H%M") + '.pkl')





