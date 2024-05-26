import pandas as pd
from itertools import product
from model.model_init import PITGAN
from data_processing.data_transformer import DataTransformer
from evaluation.privacy import compute_TCAP
from evaluation.utility import compute_CIO, compute_ROC, compute_pMSE, compute_CIO_folds


"""
This contains the training procedure for the census data
"""

def evaluate_full(X_train, transformer, discrete_columns, latent_dims, hidden_dimensions, parameters, hyper_params_sup, 
                  hyper_params_unsup, auto_epochs, gen_epochs, key_sets, regressions, utility_weights):

    # get the grid of different hyperparamters to use during training
    combinations = list(product(*hyper_params_unsup.values()))
    grid = pd.DataFrame(combinations, columns=hyper_params_unsup.keys())

    # Initialize a dict to store all of the parameters
    evaluation_results = []
    global_hyper_params = []

    # We perform the same procedure for each of the different latent dimensions
    for latent_dim in latent_dims:

        # Initialize variables to store utility scores for the latent space fitting
        sup_max_utility = 0
        max_sup_ROC = None
        sup_max_param = None
        optimal_params = hyper_params_unsup

        privacy = None
        max_ROC = None
        max_CIO = None
        max_pMSE = None

        # First we try out different using the list of hyper paramaters for the supervised learning
        if latent_dim > 0:

            for sup_grad in hyper_params_sup:

                parameters['grad_step_autoencoding'] = sup_grad

                model = PITGAN(latent_dim, hidden_dimensions, parameters, transformer)

                losses = model.fit_autoencoder(X_train, auto_epochs)

                # Compute the ROC of the decoded data to estimate the utility of the latent encoding
                decoded = model.get_decoded(X_train)
                sup_ROC = compute_ROC(X_train, decoded, transformer)

                sup_ROC_avg = sup_ROC['ROC'].mean()

                # If this latent encoding yielded a better space we pick it and update the max_param
                if sup_ROC_avg > sup_max_utility:

                    sup_max_utility = sup_ROC_avg
                    max_sup_ROC = sup_ROC_avg
                    sup_max_param = sup_grad

        # Then we select the parameters which is optimal as long as the dimension is feater than zero
        if sup_max_param is not None:
            # select the best value for autoencodeing
            parameters['grad_step_autoencoding'] = sup_max_param
            optimal_params['grad_step_autoencoding'] = sup_max_param

        # Set the initial utililty target
        max_utility = 0

        # Next we perform the unsupervised training for each of the parameters in the grid
        for index, params in grid.iterrows():

            # Set the model parameters based on the parameters selected
            for param in grid.columns:
                
                parameters[param] = params[param]

            # Initialize the model with the new parameters
            model = PITGAN(latent_dim, hidden_dimensions, parameters, transformer)

            # Fit both compoenents
            sep_losses, gen_losses = model.fit(X_train, auto_epochs, gen_epochs)

            """
            Next we perform the evaluations. Where at the end the averages of the utilities are computed to 
            determine which model is to be picked for the 
            """
            # First Generate a dataset
            X_hat = model.generate(X_train, 1)
            
            # Get the privacy
            priv = compute_TCAP(X_train, X_hat, key_sets, transformer, 0.95)

            # Compute all of the utilities
            ROCs = compute_ROC(X_train, X_hat, transformer)
            CIOs = compute_CIO_folds(X_train, model, regressions, 5)
            pMSEs = compute_pMSE(X_train, X_hat, discrete_columns)

            # Next compute the averages and then weight them together using the specified utility weights
            ROC = ROCs['ROC'].mean()
            CIO = []
            for regression in CIOs['Regression'].unique():
                CIO.append(CIOs[CIOs['Regression']==regression]['CIO'].mean())
            CIO = sum(CIO)/len(CIO)
            pMSE = pMSEs[1]

            total_utility = ROC*utility_weights[0]+CIO*utility_weights[1]+pMSE*utility_weights[2]

            # Next we all of the stored metrics if the total utility is better than the current one
            if total_utility > max_utility:

                # Update the evaluation metrics
                privacy = priv
                max_ROC = ROC
                max_CIO = CIO
                max_pMSE = pMSEs[0]
                max_pMSE4 = pMSEs[1]

                # Update the set of optimal hyper parameters
                for param in grid.columns:
                    optimal_params[param] = params[param]

                max_utility = total_utility

        # Once all of the parameters have been tested we can add the results for this latent dimension
        results = {
            'Latent_dim': latent_dim,
            'Decoded_ROC': max_sup_ROC,
            'ROC': max_ROC,
            'CIO': max_CIO,
            'pMSE': max_pMSE,
            'PMSE4': max_pMSE4,
        }
        # Add the privacy measured for each set of keys
        for keys in privacy['Num_Keys'].unique():
            results[keys + '_TCAP'] = privacy[privacy['Num_Keys']==keys]['TCAP'].mean()
            results[keys + '_TCAP_raw'] = privacy[privacy['Num_Keys']==keys]['TCAP_raw'].mean()

        # Also add the optimal parameters for this latent dim
        optimal_params_selected = {'Latent_dim': latent_dim}
        for param_selected in optimal_params.keys():
            optimal_params_selected[param_selected] = optimal_params[param_selected]

        evaluation_results.append(results)
        global_hyper_params.append(optimal_params_selected)

    # Onece all of the latent dimensions are done turn the results and paramater dicts into dataframes
    evaluation_results = pd.DataFrame(evaluation_results)
    global_hyper_params = pd.DataFrame(global_hyper_params)

    return evaluation_results, global_hyper_params
                


# Function that runs a quick evaluation on a single model
def evaluate(X_train, model, transformer, discrete_columns, utility_weights, key_sets, regressions):

    # Get the latent dimension 
    latent_dim = model.get_latent_dim()

    # Generate a base synthetic dataset
    X_hat = model.generate(X_train, 1)
         
    # Compute the privacy using the specified keys
    privacy = compute_TCAP(X_train, X_hat, key_sets, transformer, 0.95)

    # Compute the utility for the decoder
    decoded = model.get_decoded(X_train)
    sup_ROCs = compute_ROC(X_train, decoded, transformer)
    sup_ROC = sup_ROCs['ROC'].mean()

    # Compute all of the utilities
    ROCs = compute_ROC(X_train, X_hat, transformer)
    CIOs = compute_CIO_folds(X_train, model, regressions, 5)
    pMSEs = compute_pMSE(X_train, X_hat, discrete_columns)

    # Next compute the averages and then weight them together using the specified utility
    ROC = ROCs['ROC'].mean()
    CIO = []
    for regression in CIOs['Regression'].unique():
        CIO.append(CIOs[CIOs['Regression']==regression]['CIO'].mean())
    CIO = sum(CIO)/len(CIO)
    pMSE = pMSEs[1]
    total_utility = ROC*utility_weights[0]+CIO*utility_weights[1]+pMSE*utility_weights[2]

    # Initialize the output results
    results = {
        'Latent_dim': latent_dim,
        'Decoded_ROC': sup_ROC,
        'ROC': ROC,
        'CIO': CIO,
        'pMSE': pMSEs[0],
        'PMSE4': pMSEs[1],
        'Utility': total_utility
    }

    # Add the privacy measured for each set of keys
    for keys in privacy['Num_Keys'].unique():
        results[keys + '_TCAP'] = privacy[privacy['Num_Keys']==keys]['TCAP'].mean()
        results[keys + '_TCAP_raw'] = privacy[privacy['Num_Keys']==keys]['TCAP_raw'].mean()
        results[keys + '_TCAP_real'] = privacy[privacy['Num_Keys']==keys]['TCAP_real'].mean()
        results[keys + '_TCAP_raw_real'] = privacy[privacy['Num_Keys']==keys]['TCAP_raw_real'].mean()

    results = [results]

    return pd.DataFrame(results)


    



















