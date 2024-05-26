import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.losses import CategoricalCrossentropy, BinaryCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from .model_parts import build_encoder, build_decoder, build_generator, build_critic
from .model_helpers import apply_regular_softmax, apply_gumbel_softmax, gumbel_sigmoid, batch_data, batch_data_latent
from data_processing.data_transformer import DataTransformer, move_continuous, reverte_continuous

"""
This file defines the full PITGAN model, with initilization, comiling, summary, and training functions

The inputs for the model are as follows

 L : the dimensionality of the latent space

 Transformer: The transformer pretrained for the data set of interest

 Hidden dimensions:
 - dim_e : a list of length n_e where the ith entry has the number of dimensions for the ith layer in the encoder
 - dim_r : a list of length n_r where the ith entry has the number of dimensions for the ith layer in the decoder
 - dim_g : a list of length n_g where the ith entry has the number of dimensions for the ith layer in the generator
 - dim_c : a list of length n_d where the ith entry has the number of dimensions for the ith layer in the critic

Training hyperparameters
 - alpaha_sup : the weight to balance the supervised and unsupervised loss for the generator
 - alpha_grad : the weight to balance the gradient penalty and unsupervised loss
 - grad_step_autoencoding : the size of the gradient steps for the latent space training
 - grad_step_generator : the size of the gradient steps for the generator training
 - grad_step_critic : the size of the gradient steps for the critic training
 - critic_steps : number of critic updates per generator update
 - batch_size : batch sizes for training
 - sigmoid_temp : the temperature to use in the gumbel sigmoid for the encoder
 - softmax_temp : the temperature to use in the gumbel softmax for the decoder
 - scale_continuous: scaling for the continuous values,  may help critic separate continuous points
 - scale_noise: scaling the input noise for the generator, to help space out the uniform distribution
 - scale_discrete: scaling the discrete input for the critic, to help space out the discrete distributions

"""


class PITGAN(Model):

    # Takes three dicts containing all of the respected parameters listed above and uses these parameters 
    # to initialize the relevant networks
    def __init__(self, L, hidden_dimensions, parameters, transformer, *args, **kwargs):

        #superimpose the other init args
        super().__init__(*args, **kwargs)

        # Get the relevant arguments
        D_list, C_list = transformer.get_relevant_dimensions()
        R = parameters.get('R')

        dim_e = hidden_dimensions.get('dim_e')
        dim_r = hidden_dimensions.get('dim_r')
        dim_g = hidden_dimensions.get('dim_g')
        dim_c = hidden_dimensions.get('dim_c')

        # Store the lists for the discrete dimensions as well as the training parameters
        self.D_list = D_list
        self.C_list = C_list
        self.R = R
        self.L = L

        # self.alpha_sup = parameters.get('alpha_sup')
        self.alpha_grad = parameters.get('alpha_grad')
        self.alpha_sigma = parameters.get('alpha_sigma')
        self.critic_steps = parameters.get('critic_steps')
        self.batch_size = parameters.get('batch_size')
        # self.latent_sharpness = parameters.get('latent_sharpness')
        # decoder_dropout = parameters.get('decoder_dropout')
        

        self.transformer = transformer

        # Initialize the optimizers for the different gradient steps
        self.autoencoder_optimizer = Adam(learning_rate=parameters.get('grad_step_autoencoding'))
        self.generator_optimizer = Adam(learning_rate=parameters.get('grad_step_generator'))
        self.critic_optimizer = Adam(learning_rate=parameters.get('grad_step_critic'))

        # Initialize the loss functions
        self.BCE = BinaryCrossentropy(from_logits=False)
        self.CCE = CategoricalCrossentropy(from_logits=False)

        # Build the network parts
        self.encoder = build_encoder(D_list, C_list, L, dim_e)
        self.decoder = build_decoder(D_list, C_list, L, dim_r)
        self.generator = build_generator(R, L, D_list, C_list, dim_g)
        self.critic = build_critic(D_list, C_list, dim_c, L)

    # Compiles the model before taining the model
    def compile(self, *args, **kwargs):

        super().compile(*args, **kwargs)

    # Summarizes the model of all the network parts
    def get_summary(self):

        print(self.encoder.summary())
        print("###################################################################")
        print(self.decoder.summary())
        print("###################################################################")
        print(self.generator.summary())
        print("###################################################################")
        print(self.critic.summary())

    def get_latent_dim(self):

        return self.L

    """
    The next part defines the gradient steps for each network part to be used later in full training
    """

    # Gradient step for the latent autoencoder
    def train_autoencoder_step(self, X):

        # Combine the trainable variables from the encoder and decoder so they can be kept track of in the tape
        trainable_variables = self.encoder.trainable_variables + self.decoder.trainable_variables

        with tf.GradientTape() as tape:

            # Compute the latent encoding and the recovered encoding. Note that we use gumbel sigmoid in this step
            # to make the latent output "discrete" during training
            Y_hat = self.encoder(X, training = True)
            X_hat = self.decoder(Y_hat, training = True)

            # Then compute the Categorical Cross Entropy between the real data and the decoder output for each
            # categorical and mode variable separately
            start_idx = 0 
            reconstruction_loss = 0

            for dim in self.D_list + self.C_list: 
                end_idx = start_idx + dim
                loss_segment = self.CCE(X[:, start_idx:end_idx], X_hat[:, start_idx:end_idx])
                reconstruction_loss += loss_segment
                start_idx = end_idx

            # Average the loss over the number of segments (i.e., variables)
            reconstruction_loss /= len(self.D_list + self.C_list)

        # Compute the gradients an update the weights
        grads = tape.gradient(reconstruction_loss, trainable_variables)
        self.autoencoder_optimizer.apply_gradients(zip(grads, trainable_variables))

        # return the reconstruction loss
        return reconstruction_loss
    
    # Gradient step for pushing encoder output to the ends
    def train_encoder_step(self, X, Y):

        # Combine the trainable variables from the encoder and decoder so they can be kept track of in the tape
        trainable_variables = self.encoder.trainable_variables

        with tf.GradientTape() as tape:

            # Compute the latent encoding and the recovered encoding. Note that we use gumbel sigmoid in this step
            # to make the latent output "discrete" during training
            Y_hat = self.encoder(X, training = True)

            discrete_score = self.BCE(Y, Y_hat)

        # Compute the gradients an update the weights
        grads = tape.gradient(discrete_score, trainable_variables)
        self.autoencoder_optimizer.apply_gradients(zip(grads, trainable_variables))

        # return the reconstruction loss
        return discrete_score

 
    # Gradient step for the generator under supervised learning. 
    def train_generator_sup_step(self, Y):

        # Size of the batch
        batch_size = Y.shape[0]

        # Combine the trainable variables from the encoder and decoder so they can be kept track of in the tape
        trainable_variables = self.generator.trainable_variables

        with tf.GradientTape() as tape:

            # Generate a batch based on the latent values
            Z = tf.random.uniform((batch_size, self.R))
            X_hat = self.generator([Z, Y], training=True)

            # Get the latent values of the generator output
            total_discrete_dims = sum(self.D_list) + sum(self.C_list)
            X_hat_b = X_hat[:, :total_discrete_dims]
            Y_hat = self.encoder(X_hat_b)
            
            # Compute the Binary Cross Entropy between the generators latent output and the latent output it was 
            # conditioned on
            supervised_loss = self.BCE(Y, Y_hat)

        # Compute the gradients an update the weights
        grads = tape.gradient(supervised_loss, trainable_variables)
        self.generator_optimizer.apply_gradients(zip(grads, trainable_variables))

        # Return the two losses separately
        return supervised_loss
    
    
    # Gradient step for the generator. Note throughout that we try to make the optimization criteria for the generator
    # as differentiable as possible to help mitigate the complexity of the gradient computations in this step.
    def train_generator_unsup_step(self, Y):

        # Size of the batch
        batch_size = Y.shape[0]

        # Combine the trainable variables from the encoder and decoder so they can be kept track of in the tape
        trainable_variables = self.generator.trainable_variables

        with tf.GradientTape() as tape:

            # Generate a sample based on this latent dimension
            Z = tf.random.uniform((batch_size, self.R))
            X_hat = self.generator([Z, Y], training=True)

            """
            # Get the latent values of the generated samples
            total_discrete_dims = sum(self.D_list) + sum(self.C_list)
            X_hat_b = X_hat[:, :total_discrete_dims]
            X_hat_b = tf.sigmoid(self.latent_sharpness*(X_hat_b-0.5))
            Y_hat = self.encoder(X_hat_b)

            # Apply some additional shapening to the latent estimate to mimic discrete mapping
            Y_hat = tf.sigmoid(self.latent_sharpness*(Y_hat-0.5))

            # Compute the Binary Cross Entropy between the generators latent output and the latent output it was 
            # conditioned on
            supervised_loss = self.BCE(Y, Y_hat)
            """

            # Let the critic evaluate the generated data for this latent dimension
            W = self.critic([X_hat, Y])

            # Compute the expectation of the critic output to get the portion of the wasserstein loss the
            # generator needs to maximize 
            unsupervised_loss = tf.reduce_mean(W)

            # Combine the loss functions into one loss using the weight hyper parameter to balance the two losses
            generator_loss = -unsupervised_loss

        # Compute the gradients an update the weights
        grads = tape.gradient(generator_loss, trainable_variables)
        self.generator_optimizer.apply_gradients(zip(grads, trainable_variables))

        # Return the two losses separately
        return unsupervised_loss


    # Gradient step for the critic
    def train_critic_step(self, X, Y):

        # Size of the batch
        batch_size = X.shape[0]

        # Combine the trainable variables from the encoder and decoder so they can be kept track of in the tape
        trainable_variables = self.critic.trainable_variables

        with tf.GradientTape() as tape:

            # Use latent values to generate synthetic data
            Z = tf.random.uniform((batch_size, self.R))
            X_hat = self.generator([Z, Y])

            # Pass the real and generated output to the critic for the wasserstein losses, conditioning both on the 
            # given latent space
            W_real = self.critic([X, Y], training=True)
            W_generated = self.critic([X_hat, Y], training=True)

            # Next we need to get the gradients of the critic with respect to its input, and get its norm

            # sample uniform noise to combine the outputs and mix the samples
            epsilon = tf.random.uniform([batch_size, 1])
            X_mixed = epsilon * tf.dtypes.cast(X, tf.float32) + (1 - epsilon) * X_hat

            # compute the gradients of the critic with respect to the input data
            with tf.GradientTape() as gp_tape:
                gp_tape.watch(X_mixed)
                C_mixed = self.critic([X_mixed, Y], training=True)

            gp_grads = gp_tape.gradient(C_mixed, X_mixed)
            grad_norms = tf.sqrt(tf.reduce_sum(tf.square(gp_grads), axis=-1))

            # Compute the wasserstein loss as normal (again take the negative to flip)
            unsupervised_loss = tf.reduce_mean(W_real) -tf.reduce_mean(W_generated)

            # Compute the gradient penalty
            gradient_loss = tf.reduce_mean(tf.square(grad_norms - 1))

            # combine the loss functions into one loss using the weight hyper parameter to balance the two losses
            # (again negate the unsupervised component to maximize it)
            critic_loss = -unsupervised_loss + self.alpha_grad*gradient_loss

        # Compute the gradients an update the weights
        grads = tape.gradient(critic_loss, trainable_variables)
        self.critic_optimizer.apply_gradients(zip(grads, trainable_variables))

        # Return the two losses separately
        return unsupervised_loss, gradient_loss

    """
    Define the full training loops for the autoencoder and supervised training using the gradient steps defined above
    """

    # Full traing procedure for the encoder decoder training
    def fit_autoencoder(self, X_data, epochs):

        # Before training transform the data for proper use
        X_data = self.transformer.transform(X_data)
        X_data = move_continuous(X_data, self.D_list, self.C_list)

        # Get only the discrete features of X
        total_discrete_dims = sum(self.D_list) + sum(self.C_list)
        X_data = X_data[:, :total_discrete_dims]

        print(f"---Starting Autoencoder Training")

        # Array to store the average losses over each epoch
        reconstruction_losses = []

        # Iterate over all of the data in each epoch
        for epoch in range(epochs):

            # Batch the data
            batched_data = batch_data(X_data, self.batch_size)

            # Array to store all the losses in the epoch
            epoch_reconstruction_losses = []

            # Iterate through all of the batches updating the netork for each batch
            for batch in batched_data:

                # Run the gradient step
                reconstruction_loss_step = self.train_autoencoder_step(batch)

                # Add the reconstruction loss to the epoch losses
                epoch_reconstruction_losses.append(reconstruction_loss_step)

            # Compute the average losses from this epoch and add it to the overall losses
            epoch_reconstruction_loss = np.mean(epoch_reconstruction_losses)
            reconstruction_losses.append(epoch_reconstruction_loss)

            # Print the loss for this epoch
            print(f'Epoch {epoch}: Reconstruction Loss = {epoch_reconstruction_loss}')

        print(f"---Finished Autoencoder Training")

        losses = {
            "reconstruction" : reconstruction_losses
        }

        return losses
    
    # Full traing procedure for the encoder decoder training
    def fit_encoder(self, X_data, epochs):

        # Before training transform the data for proper use
        X_data = self.transformer.transform(X_data)
        X_data = move_continuous(X_data, self.D_list, self.C_list)

        # Get the latent encoding for the data
        total_discrete = sum(self.D_list) + sum(self.C_list)
        X_data = X_data[:, :total_discrete]

        Y = self.encoder(X_data)
        Y = tf.cast(tf.greater(Y, 0.5), Y.dtype)

        print(f"---Starting Encoder Training")

        # Array to store the average losses over each epoch
        discrete_losses = []

        # Iterate over all of the data in each epoch
        for epoch in range(epochs):

            # Batch the data
            batched_data = batch_data_latent(X_data, Y, self.batch_size)

            # Array to store all the losses in the epoch
            epoch_discrete_losses = []

            # Iterate through all of the batches updating the netork for each batch
            for X_batch, Y_batch in batched_data:

                # Run the gradient step
                discrete_loss_step = self.train_encoder_step(X_batch, Y_batch)

                # Add the reconstruction loss to the epoch losses
                epoch_discrete_losses.append(discrete_loss_step)

            # Compute the average losses from this epoch and add it to the overall losses
            epoch_discrete_loss = np.mean(epoch_discrete_losses)
            discrete_losses.append(epoch_discrete_loss)

            # Print the loss for this epoch
            print(f'Epoch {epoch}: Encoder Loss = {epoch_discrete_loss}')

        print(f"---Finished Encoder Training")

        losses = {
            "discrete_score" : discrete_losses
        }

        return losses
    
    # Full traing procedure for the encoder decoder training
    def fit_supervised(self, X_data, epochs):

        # Before training transform the data for proper use
        X_data = self.transformer.transform(X_data)
        X_data = move_continuous(X_data, self.D_list, self.C_list)

        # Get the latent encoding for the data
        total_discrete = sum(self.D_list) + sum(self.C_list)
        X_data = X_data[:, :total_discrete]

        Y = self.encoder(X_data)
        Y = tf.cast(tf.greater(Y, 0.5), Y.dtype)
        Y = tf.cast(Y, dtype=X_data.dtype)


        print(f"---Starting Supervised Training")

        # Array to store the average losses over each epoch
        losses = []

        # Iterate over all of the data in each epoch
        for epoch in range(epochs):

            # Batch the data
            batched_data = batch_data(Y, self.batch_size)

            # Array to store all the losses in the epoch
            epoch_losses = []

            # Iterate through all of the batches updating the netork for each batch
            for Y_batch in batched_data:

                # Run the gradient step
                supervised_loss_step = self.train_generator_sup_step(Y_batch)

                # Add the reconstruction loss to the epoch losses
                epoch_losses.append(supervised_loss_step)

            # Compute the average losses from this epoch and add it to the overall losses
            epoch_loss = np.mean(epoch_losses)
            losses.append(epoch_loss)

            # Print the loss for this epoch
            print(f'Epoch {epoch}: Supervised Loss = {epoch_loss}')

        print(f"---Finished Supervised Training")

        losses = {
            "supervised" : losses
        }

        return losses


    # Full training procedure for the generator and critic
    def fit_unsupervised(self, X_data, epochs):

        # Before training transform the data for proper use
        X_data = self.transformer.transform(X_data)
        X_data = move_continuous(X_data, self.D_list, self.C_list)

        # Get the latent encoding for the data
        num_samples = X_data.shape[0]
        total_discrete = sum(self.D_list) + sum(self.C_list)
        X_data_b = X_data[:, :total_discrete]

        Y = self.encoder(X_data_b)
        Y = tf.cast(tf.greater(Y, 0.5), Y.dtype)
        Y = tf.cast(Y, dtype=X_data.dtype)

        print(f"---Starting Unsupervised Training")

        # Array to store the average losses for all losses over each epoch
        unsupervised_generator_losses = []
        unsupervised_critic_losses = []
        gradient_penalty_losses = []
        supervised_losses = []

        # Iterate over all of the data in each epoch
        for epoch in range(epochs):

            # Get the batches for both the generator and the critic
            generator_batches = iter(batch_data(Y, self.batch_size))
            critic_iterator = iter(batch_data_latent(X_data, Y, self.batch_size))

            # Array to store all the losses in the epoch
            epoch_unsupervised_critic_losses = []
            epoch_gradient_penalty_losses = []
            epoch_unsupervised_generator_losses = []
            #epoch_supervised_losses = []

            # We iterate through all of the batches, where these batches are used only for the generator.
            # A separete iterator over the batches are then used to update the critic critic_steps number of 
            # times each time the generator is updated. This is done both so that the generator sees all of the
            # data in each epoch, and also so that the critic is being updated on different iterations of the
            # batched data throughout to avoid overfitting to any particular batch
            for batch in generator_batches:

                # Traing the critic for critic_steps number of steps
                for i in range(self.critic_steps):

                    # Try to iterate over the batch, and if not possible then we create a new set of batches
                    # that the critic can iterate over
                    try:
                        # Get the batches
                        critic_batch = next(critic_iterator)

                    except StopIteration:
                        # Update the batches and draw sample
                        critic_iterator = iter(batch_data_latent(X_data, Y, self.batch_size))
                        critic_batch = next(critic_iterator)

                    X_batch = critic_batch[0]
                    Y_batch = critic_batch[1]

                    # Run gradient step for the critic based on the new batch
                    unsupervised_critic_loss, gradient_penalty_loss = self.train_critic_step(X_batch, Y_batch)

                    # Add the losses to the epoch losses
                    epoch_unsupervised_critic_losses.append(unsupervised_critic_loss)
                    epoch_gradient_penalty_losses.append(gradient_penalty_loss)

                # Run the gradient step for the generator
                unsupervised_generator_loss = self.train_generator_unsup_step(batch)

                # Add the losses to the epoch losses
                epoch_unsupervised_generator_losses.append(unsupervised_generator_loss)

            # Compute the average losses from this epoch and add it to the overall losses
            epoch_unsupervised_generator_loss = np.mean(epoch_unsupervised_generator_losses)
            epoch_unsupervised_critic_loss = np.mean(epoch_unsupervised_critic_losses)
            epoch_gradient_penalty_loss = np.mean(epoch_gradient_penalty_losses)
            # epoch_supervised_loss = np.mean(epoch_supervised_losses)

            # Compute an epoch supervised loss estimate
            Z = tf.random.uniform((num_samples, self.R))
            X_hat= self.generator([Z, Y])
            X_hat_b = X_hat[:, :total_discrete]
            X_hat_b = tf.cast(tf.greater(X_hat_b, 0.5), X_hat_b.dtype)
            Y_hat = self.encoder(X_hat_b)
            Y_hat = tf.cast(tf.greater(Y_hat, 0.5), Y.dtype)
            supervised_loss = tf.reduce_mean(tf.abs(Y-Y_hat))

            # Append losses
            unsupervised_generator_losses.append(epoch_unsupervised_generator_loss)
            unsupervised_critic_losses.append(epoch_unsupervised_critic_loss)
            gradient_penalty_losses.append(epoch_gradient_penalty_loss)
            supervised_losses.append(supervised_loss)

            # Print the losses for this epoch
            print(f'Epoch {epoch}: Supervised Loss = {supervised_loss} '
                    f'Unsupervised Loss G = {epoch_unsupervised_generator_loss} '
                    f'Unsupervised Loss C = {epoch_unsupervised_critic_loss} '
                    f'Gradient Penalty = {epoch_gradient_penalty_loss}')

        print(f"---Finished Unsupervised Training")

        # Create the dict to hold the loss outputs

        losses = {
            "critic_unsup": unsupervised_critic_losses,
            "critic_grad": gradient_penalty_losses,
            "generator_unsup" : unsupervised_generator_losses,
            "generator_sup" : supervised_losses
        }

        return losses
    
    # Fit function to fit both of the models at once
    def fit(self, X_data, auto_epochs, gen_epochs):

        # train the autoencoder 
        auto_losses = self.fit_autoencoder(X_data, auto_epochs)

        # train the generator
        generator_losses = self.fit_unsupervised(X_data, gen_epochs)

        return auto_losses, generator_losses
    
    
    """
    Functions to perform inference once the model is trained
    """

    # Function to inference from the latent space
    def get_latent(self, X, probs):

        # Before training transform the data for proper use
        X = self.transformer.transform(X)
        X = move_continuous(X, self.D_list, self.C_list)

        # subset the discrete values
        total_discrete_dims = sum(self.D_list) + sum(self.C_list)
        X = X[:, :total_discrete_dims]

        # convert it to a tensor
        X = tf.convert_to_tensor(X, dtype=tf.float32)

        # get the encoders sigmoid output
        Y = self.encoder(X)

        if not probs:
            Y = tf.cast(tf.greater(Y, 0.5), Y.dtype)

        return Y

    # Function to inference from the encoder decoder
    def get_decoded(self, X):

        # Get the latent codes with compute enabled
        Y = self.get_latent(X, False)

        # Get logits for the latent encoding and apply regular softmax
        X_hat = self.decoder(Y)

        # Prepare to slice X_hat for argmax operations over the specified dimensions
        start_idx = 0
        argmax_samples = []
        
        # Argmax on the categorical variables
        for dim in self.D_list + self.C_list:
            # select group of one hots probabilities
            slice = X_hat[:, start_idx:start_idx + dim]
            # apply hard argmax to probabilities to get discrete one hots
            sampled_group = tf.cast(tf.equal(slice, tf.reduce_max(slice, axis=-1, keepdims=True)), dtype=tf.float32)
            # append one hots to list
            argmax_samples.append(sampled_group)
            # update start index for next group
            start_idx += dim

        # concat the one hots
        X_hat = tf.concat(argmax_samples, axis=-1)
        X_hat = X_hat.numpy()

        # Then perform an inverse transformation to get output in the original format
        X_hat = reverte_continuous(X_hat, self.D_list, self.C_list)
        X_hat = self.transformer.inverse_transform(X_hat)

        return X_hat

    # Function to inference from the generator. Note that this requires conditioning on real data. 
    # Specify k to determine the number of samples to draw from each point of data.
    def generate(self, X, k):

        # Get the latent codes with compute enabled
        Y = self.get_latent(X, False)

        # place to store output
        output_batches = []

        # get the number of samples to generate in each batch
        num_samples = X.shape[0]

        # generate for each batch
        for i in range(k):
            # get the generated output
            Z = tf.random.uniform((num_samples, self.R))
            X_tilde = self.generator([Z, Y])

            output_batches.append(X_tilde)

        # concatenate the output batches
        X_hat = tf.concat(output_batches, axis=0)
        
        # Prepare to slice X_hat for argmax operations over the specified dimensions
        start_idx = 0
        argmax_samples = []
        
        # Argmax on the categorical variables
        for dim in self.D_list + self.C_list:
            # select group of one hots probabilities
            slice = X_hat[:, start_idx:start_idx + dim]
            # apply hard argmax to probabilities to get discrete one hots
            sampled_group = tf.cast(tf.equal(slice, tf.reduce_max(slice, axis=-1, keepdims=True)), dtype=tf.float32)
            # append one hots to list
            argmax_samples.append(sampled_group)
            # update start index for next group
            start_idx += dim

        # concat the one hots
        X_hat_one_hots = tf.concat(argmax_samples, axis=-1) 

        # get the continuous parts of the generator output and concatenate them to final output
        X_hat_continuous = X_hat[:, start_idx:]

        X_hat_output = tf.concat([X_hat_one_hots, X_hat_continuous], axis=-1)

        X_hat_output = X_hat_output.numpy()

        # Then perform an inverse transformation to get output in the original format
        X_hat_output = reverte_continuous(X_hat_output, self.D_list, self.C_list)
        X_hat_output = self.transformer.inverse_transform(X_hat_output)

        return X_hat_output
    






