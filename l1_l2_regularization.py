"""
A script to compare the effect of L1 and L2 regularization on the loss curves.

author: Fabrizio Musacchio (fabriziomusacchio.com)
date: March 25, 2023

For reproducibility:

conda create -n l1_l2_regularization python=3.9
conda activate l1_l2_regularization
conda install -y mamba
mamba install -y numpy matplotlib tensorflow
"""
# %% IMPORTS
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
# %% DEFINE THE CNN ARCHITECTURE
# Load the dataset (e.g., CIFAR-10)
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0

# Define the CNN architecture
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))

# Compile the model with L1 regularization
model_l1 = models.clone_model(model)
model_l1.set_weights(model.get_weights())
model_l1.add(layers.Dense(10, kernel_regularizer=tf.keras.regularizers.l1(0.01)))

# Compile the model with L2 regularization
model_l2 = models.clone_model(model)
model_l2.set_weights(model.get_weights())
model_l2.add(layers.Dense(10, kernel_regularizer=tf.keras.regularizers.l2(0.01)))

# %% TRAIN THE MODELS
# Train the models
model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
model_l1.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
model_l2.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))
history_l1 = model_l1.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))
history_l2 = model_l2.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))
# %% PLOTS
# Plot the loss curves
plt.figure(figsize=(7, 6))
plt.plot(history.history['loss'], label='Training Loss (No Regularization)', 
         color='k', lw=2)
plt.plot(history.history['val_loss'], label='Validation Loss (No Regularization)', 
         color='k', ls='--', lw=2)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Comparison of Loss Curves (no regularization))')
plt.legend()
plt.xlim(-0.5, 9.5)
plt.ylim(0.7, 1.7)
plt.xticks(range(10))
plt.show()

plt.figure(figsize=(7, 6))
plt.plot(history_l1.history['loss'], label='Training Loss (L1 Regularization)', 
         color='#0072B2', lw=2)
plt.plot(history_l1.history['val_loss'], label='Validation Loss (L1 Regularization)', 
         color='#0072B2', ls='--', lw=2)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Comparison of Loss Curves (with L1 regularization)')
plt.legend()
plt.xlim(-0.5, 9.5)
plt.ylim(0.7, 1.7)
plt.xticks(range(10))
plt.show()

plt.figure(figsize=(7, 6))
plt.plot(history_l2.history['loss'], label='Training Loss (L2 Regularization)', 
         color='#E69F00', lw=2)
plt.plot(history_l2.history['val_loss'], label='Validation Loss (L2 Regularization)', 
         color='#E69F00', ls='--', lw=2)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Comparison of Loss Curves (with L2 regularization)')
plt.legend()
plt.xlim(-0.5, 9.5)
plt.ylim(0.7, 1.7)
plt.xticks(range(10))
plt.show()
# %% END