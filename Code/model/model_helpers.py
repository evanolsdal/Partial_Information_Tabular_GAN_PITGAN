import tensorflow as tf

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

# Application process for regular softmax for the generator output
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

    # Add gumbel noise and apply softmax

    # Add the gumbel noise and apply softmax to get the soft probabilities
    y = (logits + gumbel_noise) / temperature
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

# gets one instance of a batch from the data
def batch_data(x_train, batch_size):

    batch = tf.data.Dataset.from_tensor_slices(x_train)
    batch = batch.shuffle(buffer_size=len(x_train))
    batch = batch.batch(batch_size)
    batch = batch.prefetch(tf.data.AUTOTUNE)

    return batch