import tensorflow as tf
import numpy as np
import math
from tabulate import tabulate

"""
This section creates custom Gumbel Sigmoid and Gumbel Softmax activation functions and the procedures to 
apply these to the encoder and generator logit outputs respectively. These functions take the linear 
logit outputs of the encoder and generator and convert this output to probabilities. The reason why 
these were created separately from the network parts was to allow for the flexibility of having either 
soft and hard outputs depending on the context of how these outputs are being used during training.
The gumbeled version of the two fuction take in an extra temperature parameter which controls how 
sharp, or discrete, the outputs of this network are to mimic sampeling. 

Furthermore, both of these gumbel activations are implemented with Straight Through (ST) functionality 
to reinforece their interpretations as discrete during training, as this allows the forward pass 
through these activations to be fully discrete, while maintaining differentiability in the back prop.

Its important to note that their regular sigmoid and softmax counterparts will also need custom
application procedures, as the output from the network parts will not be segmented yet.

"""

"""
Functions used to apply the different types of softmax to the generators categorical and mode output 
to generate either probabilities or discrete outputs 
"""

# Application process for regular softmax for the generator output if output is sharp this means we
# make the output discrete
def apply_regular_softmax(generator_output, D_list, C_list):

    start_idx = 0
    processed_outputs = []

    # Process categorical and mode outputs
    for dim in D_list + C_list:
        end_idx = start_idx + dim
        softmax_output = tf.nn.softmax(generator_output[:, start_idx:end_idx], axis=-1)
        processed_outputs.append(softmax_output)
        start_idx = end_idx

    # Append the continuous part from the original generator output
    continuous_output = generator_output[:, start_idx:]
    processed_outputs.append(continuous_output)

    # Concatenate all processed outputs back together
    final_output = tf.concat(processed_outputs, axis=-1)

    return final_output

# Application process for gumbel sofrmax for the the generator output. Note that we dont include the
# continuous part in this output since its going to be passed to the encoder (which takes the discrete output)
def apply_gumbel_softmax(generator_output, D_list, C_list, temperature):

    start_idx = 0
    processed_outputs = []

    # Process categorical and mode outputs
    for dim in D_list + C_list:
        end_idx = start_idx + dim
        gumbel_output = gumbel_softmax(generator_output[:, start_idx:end_idx], temperature=temperature)
        processed_outputs.append(gumbel_output)
        start_idx = end_idx

    # Concatenate all processed outputs back together
    final_output = tf.concat(processed_outputs, axis=-1)

    return final_output


"""
This section contains the activation functions to compute the gumbeled versions of sigmoid and softmax
"""

# Custom function for sampeling gumbel noise
def sample_gumbel(shape):
    eps = 1e-20
    u = tf.random.uniform(shape, minval=0, maxval=1)
    return -tf.math.log(-tf.math.log(u + eps) + eps)


# Custom activation function for gumbel sigmoid
def gumbel_sigmoid(logits, temperature):

    # make sure inputs are non empty
    if logits is None:
        raise ValueError("Logits are None, cannot apply Gumbel sigmoid.")
    if temperature is None or temperature <= 0:
        raise ValueError("Invalid temperature value.")

    # Get the gumbel noise
    gumbel_logits_pos = sample_gumbel(tf.shape(logits))
    gumbel_logits_neg = sample_gumbel(tf.shape(logits))

    # Add the gumbel noise and apply sigmoid 
    soft_output = tf.sigmoid((logits + gumbel_logits_pos - gumbel_logits_neg) / temperature)

    # Hard thresholding in the forward pass
    hard_output = tf.cast(tf.greater(soft_output, 0.5), soft_output.dtype)
        
    # Straight-Through Estimator: use hard outputs for forward pass but soft gradients for backpropagation
    output = soft_output + tf.stop_gradient(hard_output - soft_output)

    return output

# Custom activation function for gumbel softmax with ST functionality
def gumbel_softmax(logits, temperature):

    # Get the gumbel noise
    gumbel_logits = sample_gumbel(tf.shape(logits))

    # Add the gumbel noise and apply softmax to get the soft probabilities
    y = (logits + gumbel_logits) / temperature
    soft_output = tf.nn.softmax(y)

    # Use the soft probabilities to sample hard one-hots for feed forward
    # This converts the soft probabilities into one-hot like vectors
    hard_output = tf.cast(tf.equal(soft_output, tf.reduce_max(soft_output, axis=-1, keepdims=True)), soft_output.dtype)

    # Ensure the forward pass uses hard_output, but backprop goes through soft_output
    output = soft_output + tf.stop_gradient(hard_output - soft_output)

    return soft_output

"""
Some other misc helper functions
"""

# Gets one instance of a batch from the data
def batch_data(x_train, batch_size):

    batch = tf.data.Dataset.from_tensor_slices(x_train)
    batch = batch.shuffle(buffer_size=len(x_train))
    batch = batch.batch(batch_size)
    batch = batch.prefetch(tf.data.AUTOTUNE)

    return batch

# Batch data for both X and Y at the same time
def batch_data_latent(X, Y, batch_size):

    # Convert Y to same type as X
    Y = tf.cast(Y, dtype=X.dtype)

    # Combine X and Y along the second axis
    combined_data = tf.concat([X, Y], axis=1)

    # Create a TensorFlow dataset from the combined data
    dataset = tf.data.Dataset.from_tensor_slices(combined_data)
    dataset = dataset.shuffle(buffer_size=len(X))
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    # Function to split combined data back into X and Y
    def split_data(batch):
        x_batch = batch[:, :X.shape[1]]
        y_batch = batch[:, X.shape[1]:]
        return x_batch, y_batch

    # Map the split function to the dataset
    dataset = dataset.map(split_data)

    return dataset


# Noise generate noise proportially for the number of discrete and continuous variables
def get_custom_noise(D_list, C_list, batch):

    discrete_noise = tf.random.uniform((batch, len(D_list)))

    continuous_noise = tf.random.normal((batch, len(C_list)))

    noise = tf.concat([discrete_noise, continuous_noise], axis=-1)

    return noise

# Function to somewhat understand what latent dimensions would be appropriate for training the model
def get_latent_dims(D_list, C_list):
    # Calculate the product of dimensions
    total_arrangements = math.prod(D_list + C_list)

    # List of tuples to store the latent dimension, number of states covered, and the number of states remaining
    info = [("Latent Dim", "States Covered", "States Remaining")]
    info.append((0, 0, total_arrangements))

    # Compute arrangements of the latent space, starting with 2 and increasing increments by two
    latent_arrangements = 2
    latent_dim = 1

    while total_arrangements > 0:
        total_arrangements -= latent_arrangements
        info.append((latent_dim, latent_arrangements, max(0, total_arrangements)))
        latent_arrangements *= 2
        latent_dim += 1

    # Print using tabulate
    print(tabulate(info, headers="firstrow", tablefmt="grid"))



