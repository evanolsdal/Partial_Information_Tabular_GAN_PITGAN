import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.losses import CategoricalCrossentorpy, BinaryCrossentropy
from tensorflow.keras.optimizers import Adam
from model_parts import build_encoder, build_decoder, build_generator, build_critic
from model_helpers import apply_regular_softmax, apply_gumbel_softmax, batch_data

"""
This file defines the full PITGAN model, with initilization, comiling, summary, and training functions

The inputs for the model are as follows

Model dimensions:
 - D_list : a list of dimensions sizes for the categorical variables, where the ith entry is the number of categories in the ith variable
 - C_list : a list of mode counts for the continuous variables, where the ith entry is the number of modes in the ith variable
 - L : the dimensionality of the latent space
 - R : the dimensions of the noise input

 Hidden dimensions:
 - dim_e : a list of length n_e where the ith entry has the number of dimensions for the ith layer in the encoder
 - dim_r : a list of length n_r where the ith entry has the number of dimensions for the ith layer in the decoder
 - dim_g : a list of length n_g where the ith entry has the number of dimensions for the ith layer in the generator
 - dim_d : a list of length n_d where the ith entry has the number of dimensions for the ith layer in the critic

Training hyperparameters
 - alpaha_sup : the weight to balance the supervised and unsupervised loss for the generator
 - alpha_grad : the weight to balance the gradient penalty and unsupervised loss
 - grad_step_autoencoding : the size of the gradient steps for the latent space training
 - grad_step_gradieny : the size of the gradient steps for the generator training
 - grad_step_critic : the size of the gradient steps for the critic training
 - batch_size : batch sizes for training
 - sigmoid_temp : the temperature to use in the gumbel sigmoid for the encoder
 - softmax_temp : the temperature to use in the gumbel softmax for the decoder
 - critic_steps : the number of steps the critic should in each loop before updating the generator

"""


class PITGAN(Model):

    # Takes three dicts containing all of the respected parameters listed above and uses these parameters 
    # to initialize the relevant networks
    def __init__(self, model_dimensions, hidden_dimensions, training_parameters, *args, **kwargs):

        #superimpose the other init args
        super().__init__(*args, **kwargs)

        # Get the relevant arguments
        D_list = model_dimensions.get('D_list')
        C_list = model_dimensions.get('C_list')
        L = model_dimensions.get('L')
        R = model_dimensions.get('R')

        dim_e = hidden_dimensions.get('dim_e')
        dim_r = hidden_dimensions.get('dim_r')
        dim_g = hidden_dimensions.get('dim_g')
        dim_c = hidden_dimensions.get('dim_c')

        # Store the lists for the discrete dimensions as well as the training parameters
        self.D_list = D_list
        self.C_list = C_list

        self.alpaha_sup = training_parameters.get('alpaha_sup')
        self.alpha_grad = training_parameters.get('alpha_grad')
        self.batch_size = training_parameters.get('batch_size')
        self.sigmoid_temp  = training_parameters.get('sigmoid_temp ')
        self.softmax_temp = training_parameters.get('softmax_temp')
        self.critic_steps = training_parameters.get('critic_steps')

        # Initialize the optimizers for the different gradient steps
        self.autoencoder_optimizer = Adam(learning_rate=training_parameters.get('grad_step_autoencoding'))
        self.generator_optimizer = Adam(learning_rate=training_parameters.get('grad_step_generator'))
        self.critic_optimizer = Adam(learning_rate=training_parameters.get('grad_step_critic'))

        # Initialize the loss functions
        self.BCE = BinaryCrossentropy(from_logits=False)
        self.CCE = CategoricalCrossentorpy(from_logits=False)

        # Build the network parts
        self.encoder = build_encoder(D_list, C_list, L, dim_e)
        self.decoder = build_decoder(D_list, C_list, L, dim_r)
        self.generator = build_generator(R, L, D_list, C_list, dim_g)
        self.critic = build_critic(D_list, C_list, dim_d, L)

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

    """
    The next part defines the gradient steps for each network part to be used later in full training
    """

    # Gradient step for the latent autoencoder
    def train_autoencoder_step(self, X):

        # Get only the discrete features of X
        total_discrete_dims = sum(self.D_list) + sum(self.C_list)
        X = X[:, :total_discrete_dims]

        # Combine the trainable variables from the encoder and decoder so they can be kept track of in the tape
        trainable_variables = self.encoder.trainable_variables + self.decoder.trainable_variables

        with tf.GradientTape() as tape:

            # Compute the latent encoding and the recovered encoding. Note that we use gumbel sigmoid in this step
            # to make the latent output "discrete" during training
            Y_hat = self.encoder(X, training = True)
            Y_hat = gumbel_sigmoid(Y_hat, self.sigmoid_temp)

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

    # Gradient step for the generator. Note throughout that we try to make the optimization criteria for the generator
    # as differentiable as possible to help mitigate the complexity of the gradient computations in this step.
    def train_generator_step(self, X):

        # Combine the trainable variables from the encoder and decoder so they can be kept track of in the tape
        trainable_variables = self.generator.trainable_variables

        with tf.GradientTape() as tape:

            # Get only the discrete features of X
            total_discrete_dims = sum(self.D_list) + sum(self.C_list)
            X_b = X[:, :total_discrete_dims]

            # Get the latent variables from the encoder from the real samples. Note that we pass the output through the 
            # regular sigmoid activation and make it discrete as we are now drawing actual samples from this latent space
            Y = self.encoder(X_b)
            Y = tf.nn.sigmoid(Y)
            Y = tf.cast(tf.greater(Y, 0.5), Y.dtype)

            # Use these as inputs to the generator, as well as some sampeled noise, 
            Z = tf.random.uniform((self.batch_size, self.R))
            X_hat = self.generator(Z, Y, training=True)

            # Compute the estimated latent space of this output. Note that we need to use gumbel softmax
            # on the generator output to simulate discrete sampeling. We also apply regular sigmoid to the output of the 
            # encoder for the latent variables, as this will make for a smoother optimization criteria with respect 
            # to the generator parameters for the BCE loss later on
            X_hat_b = apply_gumbel_softmax(X_hat, self.D_list, self.C_list, self.softmax_temp)
            Y_hat = self.encoder(X_hat_b)
            Y_hat = tf.nn.sigmoid(Y_hat)

            # Pass the latent conditioning and the generator output to the critic for the wasserstein losses. Note
            # that we first need to pass the generator output through a regular softmax. We choose regular softmax
            # of gumbel softmax since this will create smoother probability outputs for the wasserstain approximation
            X_hat = apply_regular_softmax(X_hat, self.D_list, self.C_list)
            W = self.critic(X_hat, Y)

            # Compute the Binary Cross Entropy between the generators latent output and the latent output it was 
            # conditioned on
            supervised_loss = self.BCE(Y, Y_hat)

            # Compute the expectation of the critic output to get the portion of the wasserstein loss the
            # generator needs to maximize 
            unsupervised_loss = tf.reduce_mean(W)

            # Combine the loss functions into one loss using the weight hyper parameter to balance the two losses
            # (note that unsupervised loss is negated in order to maximize it when performing gradient descent)
            generator_loss = -unsupervised_loss + self.alpha_sup*supervised_loss

        # Compute the gradients an update the weights
        grads = tape.gradient(generator_loss, trainable_variables)
        self.generator_optimizer.apply_gradients(zip(grads, trainable_variables))

        # Return the two losses separately
        return unsupervised_loss, supervised_loss


    # Gradient step for the critic
    def train_critic_step(self, X):

        # Combine the trainable variables from the encoder and decoder so they can be kept track of in the tape
        trainable_variables = self.critic.trainable_variables

        with tf.GradientTape() as tape:

            # Get only the discrete features of X
            total_discrete_dims = sum(self.D_list) + sum(self.C_list)
            X_b = X[:, :total_discrete_dims]

            # Get the latent variables from the encoder from the real samples. Note that we pass the output through the 
            # regular sigmoid activation and make it discrete as we are now drawing actual samples from this latent space
            Y = self.encoder(X_b)
            Y = tf.nn.sigmoid(Y)
            Y = tf.cast(tf.greater(Y, 0.5), Y.dtype)

            # Use these as inputs to the generator, as well as some sampeled noise, where we apply regular softmax to the
            # output of the generator to get the continuous range of probabilities
            Z = tf.random.uniform((self.batch_size, self.R))
            X_hat = self.generator(Z, Y)
            X_hat = apply_regular_softmax(X_hat, self.D_list, self.C_list)

            # Pass the real and generated output to the critic for the wasserstein losses, conditioning both on the 
            # given latent space
            W_real = self.critic(X, Y, training=True)
            W_generated = self.critic(X_hat, Y, training=True)

            # Next we need to get the gradients of the critic with respect to its input, and get its norm

            # sample uniform noise to combine the outputs and mix the samples
            epsilon = tf.random.uniform([self.batch_size, 1])
            X_mixed = epsilon * tf.dtypes.cast(X, tf.float32) + (1 - epsilon) * X_hat

            # compute the gradients of the critic with respect to the input data
            with tf.GradientTape() as gp_tape:
                gp_tape.watch(X_mixed)
                C_mixed = self.critic(X_mixed, Y, training=True)

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

        print(f"Starting Autoencoder Training")

        # Array to store the average losses over each epoch
        losses = []

        # Iterate over all of the data in each epoch
        for epoch in range(epochs):

            # Batch the data
            batched_data = batch_data(X_data, self.batch_size)

            # Array to store all the losses in the epoch
            epoch_losses = []

            # Iterate through all of the batches updating the netork for each batch
            for batch in batched_data:

                # Run the gradient step
                reconstruction_loss_step = self.train_autoencoder_step(batch)

                # Add the reconstruction loss to the epoch losses
                epoch_losses.append(reconstruction_loss_step)

            # Compute the average losses from this epoch and add it to the overall losses
            epoch_loss = np.mean(epoch_losses)
            losses.append(epoch_loss)

            # Print the loss for this epoch
            print(f'Epoch {epoch}: Reconstruction Loss = {epoch_loss}')

        print(f"Finished Autoencoder Training")

        return losses

    # Full training procedure for the generator and critic
    def fit_generative(self, X_data, epochs):

        print(f"Starting Autoencoder Training")

        # Array to store the average losses for all losses over each epoch
        unsupervised_generator_losses = []
        unsupervised_critic_losses = []
        gradient_penalty_losses = []
        supervised_losses = []

        # Iterate over all of the data in each epoch
        for epoch in range(epochs):

            # Get the batches for both the generator and the critic
            generator_batches = iter(batch_data(X_data, self.batch_size))
            critic_iterator = iter(batch_data(X_data, self.batch_size))

            # Array to store all the losses in the epoch
            epoch_unsupervised_critic_losses = []
            epoch_gradient_penalty_losses = []
            epoch_unsupervised_generator_losses = []
            epoch_supervised_losses = []

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
                        critic_iterator = iter(batch_data(X_data, self.batch_size))
                        critic_batch = next(critic_iterator)

                    # Run gradient step for the critic based on the new batch
                    unsupervised_critic_loss, gradient_penalty_loss = self.train_critic(critic_batch)

                    # Add the losses to the epoch losses
                    epoch_unsupervised_critic_losses.append(unsupervised_critic_loss)
                    epoch_gradient_penalty_losses.append(gradient_penalty_loss)

                # Run the gradient step for the generator
                unsupervised_generator_loss, supervised_loss = self.train_generator_step(batch)

                # Add the losses to the epoch losses
                epoch_unsupervised_generator_losses.append(unsupervised_generator_loss)
                epoch_supervised_losses.append(supervised_loss)

            # Compute the average losses from this epoch and add it to the overall losses
            epoch_unsupervised_generator_loss = np.mean(epoch_unsupervised_generator_losses)
            epoch_unsupervised_critic_loss = np.mean(epoch_unsupervised_critic_losses)
            epoch_gradient_penalty_loss = np.mean(epoch_gradient_penalty_losses)
            epoch_supervised_loss = np.mean(epoch_supervised_losses)

            unsupervised_generator_losses.append(epoch_unsupervised_generator_loss)
            unsupervised_critic_losses.append(epoch_unsupervised_critic_loss)
            gradient_penalty_losses.append(epoch_gradient_penalty_loss)
            supervised_losses.append(epoch_supervised_loss)

            # Print the loss for this epoch
            print(f'Epoch {epoch}: Supervised Loss = {epoch_supervised_loss} \
                                    Unsupervised Loss G = {epoch_unsupervised_generator_loss}\
                                    Unsupervised Loss C = {epoch_unsupervised_critic_loss}\
                                    Gradient Penalty = {epoch_gradient_penalty_loss}')

        print(f"Finished Autoencoder Training")

        return epoch_unsupervised_critic_losses, epoch_gradient_penalty_losses, \
                epoch_unsupervised_generator_losses, epoch_supervised_losses



