import tensorflow as tf
from tensorflow.keras import layers, Model


"""
    Build an encoder model with a linear output for the latent space. Since the inputs are discrete we use an 
    imbedding layer, represented by a dense linear activation separately on the nodes corresponding the the one 
    hots of each variable.  We output to a continuous latent space to allow for a rich latent representation.
    
    Parameters:
    - D_list: List containing the number of categories for each categorical variable.
    - C_list: List containing the number of modes for each continuous variable.
    - L: The dimensionality of the latent space to which the input data is encoded.
    
    """
def build_encoder(D_list, C_list, L, dim_e):
    
    # Assuming the input dimensions are for the concatenated one-hot encoded categorical variables and modes
    total_input_dims = sum(D_list) + sum(C_list)
    
    # Input layer for the concatenated vector of all discrete features
    input_layer = layers.Input(shape=(total_input_dims,), name='encoder_input')

    # Initialize a list to hold embeddings
    embeddings = []

    # Start index for slicing the input for each categorical variable/mode
    start_idx = 0

    # Embeddings for categorical variables
    for i, d_i in enumerate(D_list):

        cat_slice = layers.Lambda(lambda x, start, end: x[:, start:end],
                                output_shape=(d_i,),
                                arguments = {'start': start_idx, 'end': start_idx + d_i},
                                name=f'cat_slice_{i+1}')(input_layer)
        cat_embed = layers.Dense(d_i-1, activation='linear', name=f'cat_embed_{i+1}')(cat_slice)
        embeddings.append(cat_embed)
        start_idx += d_i

    # Embeddings for modes of continuous variables
    for i, m_i in enumerate(C_list):

        mode_slice = layers.Lambda(lambda x, start, end: x[:, start:end],
                                output_shape=(m_i,),
                                arguments = {'start': start_idx, 'end': start_idx + m_i},
                                name=f'mode_slice_{i+1}')(input_layer)
        mode_embed = layers.Dense(m_i-1, activation='linear', name=f'mode_embed_{i+1}')(mode_slice)
        embeddings.append(mode_embed)
        start_idx += m_i


    # Combine all embeddings
    x = layers.Concatenate(name = 'embed_concat')(embeddings)

    # Process combined embeddings through dense hidden layers as dictated by dim_e
    for i, size in enumerate(dim_e):
        x = layers.Dense(size, activation='relu', name=f'hidden_layer{i+1}')(x)
    
    # The final layer is linear to create logits that can later be used for the sigmoid activations
    latent_output = layers.Dense(L, activation='sigmoid', name='output_sigmoid')(x)
    
    # Define the encoder model
    encoder = Model(inputs=input_layer, outputs=latent_output, name="encoder")
    
    return encoder


"""
    Build a decoder model with softmax activation for the outputs of categorical variables.
    
    Parameters:
    - D_list: List containing the number of categories for each categorical variable.
    - C_list: List containing the number of modes for each continuous variable.
    - L: The dimensionality of the latent space from which the decoder starts.
    - dim_r: List containing the dimensions for each hidden layer in the decoder.
    """
def build_decoder(D_list, C_list, L, dim_r):
    
    
    # Input layer from the latent space
    latent_input = layers.Input(shape=(L,), name='input_latent')

    # Dense layers processing the latent representation
    x = latent_input
    for i, size in enumerate(dim_r):
        x = layers.Dense(size, activation='relu', name=f'hidden_layer_{i+1}')(x)
        # x = layers.Dropout(dropout, name=f'dropout_layer_{i+1}')(x)

    # Prepare output layers for both categorical and continuous variables
    output_layers = []

    # Output for categorical variables using Dense layer with softmax activation
    for i, d_i in enumerate(D_list):
        cat_output = layers.Dense(d_i, activation='softmax', name=f'cat_output_{i+1}')(x)
        output_layers.append(cat_output)

    # Output for continuous variables using Dense layer with softmax activation
    for i, m_i in enumerate(C_list):
        cat_output = layers.Dense(m_i, activation='softmax', name=f'mode_output_{i+1}')(x)
        output_layers.append(cat_output)

    # Combine all outputs 
    output_layer = layers.Concatenate(name = 'output_concat')(output_layers)

    # Define the decoder model
    decoder = Model(inputs=latent_input, outputs=output_layer, name="decoder")
    
    return decoder


"""
    Builds the generator part of the model, which takes noise and the latent code and maps it to probability 
    outputs for the categorical variables and the modes of the continuous variables, and to a constrained 
    output between -3 and 3 for the continuous part of the representation.
    
    Arguments:
    - R: Dimension of the noise input
    - L: Dimensionality of the latent space
    - D_list: List containing the number of categories for each categorical variable
    - C_list: List containing the number of modes for each continuous variable
    - dim_g: List containing the dimensions for each hidden layer
"""

def build_generator(R, L, D_list, C_list, dim_g):
    
    # Inputs
    input_noise = layers.Input(shape=(R,), name = 'input_noise')
    input_latent = layers.Input(shape=(L,), name = 'input_latent')

    # Combine to form one input
    x = layers.Concatenate(name = 'input_concat')([input_noise, input_latent])

    total_discrete_dims = sum(D_list) + sum(C_list)

    # Iteratively add the shared dense layers as dictated by dim_g up 
    for i, size in enumerate(dim_g):
        x = layers.Concatenate(name = f'hidden_concat_{i+1}')([x, input_latent])
        x = layers.Dense(size, activation='relu', name=f'hidden_layer_{i+1}')(x)

    # Prepare output layers for both categorical and continuous variables
    discrete_output = []

    # Output for categorical variables using Dense layer with softmax activation
    for i, d_i in enumerate(D_list):
        cat_output = layers.Dense(d_i, activation='softmax', name=f'cat_output_{i+1}')(x)
        discrete_output.append(cat_output)

    # Output for continuous variables using Dense layer with softmax activation
    for i, m_i in enumerate(C_list):
        cat_output = layers.Dense(m_i, activation='softmax', name=f'mode_output_{i+1}')(x)
        discrete_output.append(cat_output)

    # Output the continuous parts using tanh, add extra hidden layer for the continuous output
    continuous_output = layers.Dense(len(C_list), activation='tanh', name='output_continuous')(x)

    # Combine all outputs
    output_layer = layers.Concatenate(name = 'output_concat')([*discrete_output, continuous_output])

    # Define and return the generator model
    generator = Model(inputs=[input_noise, input_latent], outputs=output_layer, name="generator")
    
    return generator


"""
    Build a critic model to evaluate both generated and real data. The process begins by creating an 
    embedding layer for each distinct feature of the model, enabling the learning of separate representations 
    for each variable involved before combining all the information in the dense layers. This approach emphasizes 
    to the model which input groups represent the same variable; for example, the discrete one-hot vectors 
    for a categorical variable are first grouped together and then passed on to the dense layers. Similarly, 
    the modes and their continuous counterparts are grouped to achieve a similar organization of relevant 
    information. Once passed through the hidden dense layers the model outputs a single real value used 
    for the approximation of the Wasserstein distance.
    
    - D_list: List containing the number of categories for each categorical variable
    - C_list: List containing the number of modes for each continuous variable
    - dim_d: List containing the dimensions for each hidden layer
    - L: Dimensionality of the latent space
"""
def build_critic(D_list, C_list, dim_d, L):
    
    D = sum(D_list)
    C = sum(C_list)
    c = len(C_list)

    total_input_dim = D + C + c
    total_discrete_dim = D + C

    # Single concatenated input from P_r
    input_raw = layers.Input(shape=(total_input_dim,), name='input_raw')
    
    # Extract the continuous and discrete inputs separately
    input_continuous = layers.Lambda(lambda x: x[:, total_discrete_dim:],
                                    name='input_discrete_slice')(input_raw)
    
    # Additional input for the latent space P_l
    input_latent = layers.Input(shape=(L,), name='input_latent')

    # Process categorical variables
    categorical_embeddings = []
    start_idx = 0
    for i, d_i in enumerate(D_list):
        cat_slice = layers.Lambda(lambda x, start, end: x[:, start:end],
                                output_shape=(d_i,),
                                arguments = {'start': start_idx, 'end': start_idx + d_i},
                                name=f'cat_slice_{i+1}')(input_raw)
        cat_dense = layers.Dense(d_i-1, activation='linear', name=f'cat_embed_{i+1}')(cat_slice)
        categorical_embeddings.append(cat_dense)
        start_idx += d_i


    # Process modes for continuous variables and the continuous variables themselves
    continuous_embeddings = []
     # Calculate the starting index for the continuous part
    start_continuous_idx = D + C
    for i, m_i in enumerate(C_list):
        # Get mode slice
        mode_slice = layers.Lambda(lambda x, start, end: x[:, start:end],
                                    output_shape=(m_i,),
                                    arguments = {'start': start_idx, 'end': start_idx + m_i},
                                    name=f'mode_slice_{i+1}')(input_raw)
        # Get continuous variable slice
        continuous_slice = layers.Lambda(lambda x, start, end: x[:, start:end],
                                            output_shape=(1,),
                                            arguments = {'start': start_continuous_idx + i, 'end': start_continuous_idx + i+1},
                                            name=f'cont_slice_{i+1}')(input_raw)
        combined_slice = layers.Concatenate(name=f'continuous_concat_{i+1}')([mode_slice, continuous_slice])
        mode_dense = layers.Dense(m_i, activation='linear', name=f'mode_embed_{i+1}')(combined_slice)
        continuous_embeddings.append(mode_dense)
        start_idx += m_i


    # Combine all processed parts
    combined_input = layers.Concatenate(name = 'input_concat')([*categorical_embeddings, *continuous_embeddings, input_continuous, input_latent])

    # # Iteratively add the shared dense layers as dictated by dim_d up 
    x = combined_input
    for i, size in enumerate(dim_d):
        x = layers.Concatenate(name = f'hidden_concat_{i+1}')([x, input_latent])
        x = layers.Dense(size, activation='leaky_relu', name=f'hidden_layer_{i+1}')(x)

    # Output layer as real value
    output_layer = layers.Dense(1, activation='linear', name='output')(x)

    # Define and return the critic model
    critic = Model(inputs=[input_raw, input_latent], outputs=output_layer, name="critic")
    
    return critic



