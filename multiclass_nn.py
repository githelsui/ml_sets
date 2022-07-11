import tensorflow as tf
from tensorflow.python.keras import layers
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

# --- 0. Clean up outputs for results

# The following lines adjust the granularity of reporting. 
pd.options.display.max_rows = 10
pd.options.display.float_format = "{:.1f}".format

# The following line improves formatting when ouputting NumPy arrays.
np.set_printoptions(linewidth = 200)

# --- 1. Handle and Load the Dataset
# returns a dataset/arraylist per each 4 data structures
(x_train, y_train),(x_test, y_test) = tf.keras.datasets.mnist.load_data()

# What does the dataset look like?
# Ordinal names to access different subsets of the MNIST dataset
# Each example is a 3-dimensional matrix indicating the pixels 
# and the color intensity per each pixel index

# -- A. Visualize dataset
print("--> Output example #2917 of the training set.\n")
print(x_train[2917])

print("--> Visualize example #2917 of the numeric array as an image.\n")
plt.imshow(x_train[2917])
plt.show()

print("--> Output row #10 of example #2917\n") # indicates the color intesities per each pixel in row
print(x_train[2917][10])

print("--> Output pixel #16 of row #10 of example 2917\n") # indicates the color intesity of this specific pixel
print(x_train[2917][10][16])

# --- 2. Normalize: Convert raw data to an average normal value
# Scale of 0 to 255 for representing color intensity
x_train_normalized = x_train / 255.0
x_test_normalized = x_test / 255.0

print("--> Output Normalized Row from x_train\n")
print(x_train_normalized[2900][10]) # Output a normalized row

# --- 3. Classification Functions (Different from usual regression problems)
# Accuracy Curve
def plot_curve(epochs, hist, list_of_metrics):
  """Plot a curve of one or more classification metrics vs. epoch."""  
  # list_of_metrics should be one of the names shown in:
  # https://www.tensorflow.org/tutorials/structured_data/imbalanced_data#define_the_model_and_metrics  

  plt.figure()
  plt.xlabel("Epoch")
  plt.ylabel("Value")

  for m in list_of_metrics:
    x = hist[m]
    plt.plot(epochs[1:], x[1:], label=m)

  plt.legend()

print("Loaded the plot_curve function.")

# --- 4. Define the neural network model and training function
# Layers, nodes, and regularization layers, and activation functions

def create_model(my_learning_rate):
  """Create and compile a deep neural net."""
  
  # All models in this course are sequential.
  model = tf.keras.models.Sequential()

  # The features are stored in a two-dimensional 28X28 array. 
  # Flatten that two-dimensional array into a a one-dimensional 
  # 784-element array.

  # Input layer
  model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))

  # Define the first hidden layer.   
  model.add(tf.keras.layers.Dense(units=32, activation='relu'))
  
  # Define a dropout regularization layer. 
  model.add(tf.keras.layers.Dropout(rate=0.2))

  # Define the output layer. The units parameter is set to 10 because
  # the model must choose among 10 possible output values (representing
  # the digits from 0 to 9, inclusive).
  #
  # Don't change this layer.
  model.add(tf.keras.layers.Dense(units=10, activation='softmax'))     
                           
  # Construct the layers into a model that TensorFlow can execute.  
  # Notice that the loss function for multi-class classification
  # is different than the loss function for binary classification.  
  model.compile(optimizer=tf.keras.optimizers.Adam(lr=my_learning_rate),
                loss="sparse_categorical_crossentropy",
                metrics=['accuracy'])
  
  return model    


def train_model(model, train_features, train_label, epochs,
                batch_size=None, validation_split=0.1):
  """Train the model by feeding it data."""

  history = model.fit(x=train_features, y=train_label, batch_size=batch_size,
                      epochs=epochs, shuffle=True, 
                      validation_split=validation_split)
 
  # To track the progression of training, gather a snapshot
  # of the model's metrics at each epoch. 
  epochs = history.epoch
  hist = pd.DataFrame(history.history)

  return epochs, hist    

# --- 5. Training for Neural Net Classification
# The following variables are the hyperparameters.
learning_rate = 0.003
epochs = 50
batch_size = 4000
validation_split = 0.2

# Establish the model's topography.
my_model = create_model(learning_rate)

# Train the model on the normalized training set.
epochs, hist = train_model(my_model, x_train_normalized, y_train, 
                           epochs, batch_size, validation_split)

# Plot a graph of the metric vs. epochs.
list_of_metrics_to_plot = ['accuracy']
plot_curve(epochs, hist, list_of_metrics_to_plot)

# Evaluate against the test set.
print("\n Evaluate the new model against the test set:")
my_model.evaluate(x=x_test_normalized, y=y_test, batch_size=batch_size)