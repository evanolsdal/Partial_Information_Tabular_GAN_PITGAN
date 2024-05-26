import pandas as pd
import numpy as np
import statsmodels.api as sm

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error

from evaluation.evaluation_helpers import calculate_features

"""
This file contains all of the measures of utility used
"""

# Calculation of the Ration of Counts
def compute_ROC(data_real_input, data_synth_input, transformer):

    data_real = data_real_input.copy()
    data_synth = data_synth_input.copy()

    # First transform data to the mode representation of continuous variables
    data_real = transformer.transform_modes(data_real)
    data_synth = transformer.transform_modes(data_synth)

    # Identify discrete variables
    discrete_variables = data_real.columns.tolist()
    
    # Initialize ROC sum
    roc_sum = 0

    # initialize a list to hold the ROC of the individual variables
    results_list = []
    
    # Loop through each discrete variable
    for var in discrete_variables:

        # Get the frequency of each category for the variable in both datasets
        count_real = data_real[var].value_counts(normalize=True)
        count_synth = data_synth[var].value_counts(normalize=True)
        
        # Get the union of categories appearing in both datasets
        all_categories = count_real.index.union(count_synth.index)
        total_categories = len(all_categories)
        
        # Skip processing if the target column has only one unique value in either dataset
        if total_categories <= 1:
            continue

        # Initialize the sum for this variable
        var_sum = 0
        
        # Calculate r_{k_j} for each category
        for category in all_categories:
            # Proportion in real data or 0 if category is not present
            y_o_kj = count_real.get(category, 0)
            # Proportion in synthetic data or 0 if category is not present
            y_s_kj = count_synth.get(category, 0)
            
            # Calculate the ratio of the minimum to maximum proportion
            r_kj = min(y_o_kj, y_s_kj) / max(y_o_kj, y_s_kj) if max(y_o_kj, y_s_kj) > 0 else 0
            
            # Sum up the ratios for this variable
            var_sum += r_kj
        
        # Compute the average ROC score for this variable
        var_ROC = var_sum / total_categories
        
        # Append the result for this target to the list
        results_list.append({'Variable': var, 'ROC': var_ROC})

        # Add the sum for this variable to the overall ROC sum
        roc_sum += var_ROC

    # Convert list of results to a DataFrame
    results_df = pd.DataFrame(results_list)
    
    return results_df


# Function to calculate the confidence interval overalps
def compute_CIO(data_original_input, data_synthetic_input, dependent, independent_continuous, independent_discrete, alpha=0.05):

    data_original = data_original_input.copy()
    data_synthetic = data_synthetic_input.copy()

    # First subset the columns which will actually be used in the analysis and split it into the dependent and 
    # independent variables, where the independent variables are further split into the continuous and discrete parts
    data_o_Y = data_original[dependent].copy()
    data_s_Y = data_synthetic[dependent].copy()

    data_o_X_c = data_original[independent_continuous].copy()
    data_s_X_c = data_synthetic[independent_continuous].copy()

    data_o_X_d = data_original[independent_discrete].copy()
    data_s_X_d = data_synthetic[independent_discrete].copy()

    # Identify all unique categories for each categorical variable across both datasets
    for column in independent_discrete:
        combined_categories = pd.concat([data_o_X_d[column], data_s_X_d[column]]).astype('category')
        categories = combined_categories.cat.categories

        # Convert to dummies with consistent categories
        data_o_X_d[column] = pd.Categorical(data_o_X_d[column], categories=categories)
        data_s_X_d[column] = pd.Categorical(data_s_X_d[column], categories=categories)

    data_o_X_d = pd.get_dummies(data_o_X_d, drop_first=True).astype(int)
    data_s_X_d = pd.get_dummies(data_s_X_d, drop_first=True).astype(int)

    # Combine the continuous and aligned dummy variables for both datasets
    data_o_X = pd.concat([data_o_X_c, data_o_X_d], axis=1)
    data_s_X = pd.concat([data_s_X_c, data_s_X_d], axis=1)

    # Then perform the OLS regression on both datasets
    data_o_X = sm.add_constant(data_o_X) 
    model_o = sm.OLS(data_o_Y, data_o_X).fit()

    data_s_X = sm.add_constant(data_s_X) 
    model_s = sm.OLS(data_s_Y, data_s_X).fit()

    # Extract confidence intervals from both models
    ci_o = model_o.conf_int(alpha=alpha)
    ci_s = model_s.conf_int(alpha=alpha)

    # Compute the overlap of the confidence intervals for each coefficient
    overlaps = []
    common_columns = ci_o.index.intersection(ci_s.index)
    for col in common_columns:
        # Get the intervals
        lower_bound = max(ci_o.loc[col, 0], ci_s.loc[col, 0])
        upper_bound = min(ci_o.loc[col, 1], ci_s.loc[col, 1])
        overlap_width = max(0, upper_bound - lower_bound)
        total_width_o = ci_o.loc[col, 1] - ci_o.loc[col, 0]
        total_width_s = ci_s.loc[col, 1] - ci_s.loc[col, 0]
        relative_overlap_o = overlap_width / total_width_o if total_width_o > 0 else 0
        relative_overlap_s = overlap_width / total_width_s if total_width_s > 0 else 0

        if total_width_o == 0 and total_width_s == 0:
            # Both confidence intervals are points with no width, implying perfect overlap
            CIO_k = 1
        else:
            relative_overlap_o = overlap_width / total_width_o if total_width_o > 0 else 0
            relative_overlap_s = overlap_width / total_width_s if total_width_s > 0 else 0
            CIO_k = (relative_overlap_o + relative_overlap_s) / 2

        # Add the results
        overlaps.append({'Variable': col, 'CIO': CIO_k})

    return pd.DataFrame(overlaps)

# Function which performs CIO on multiple draws to balance out variance, on multiple regressions
def compute_CIO_folds(data_original, model, regressions, folds):

    results = []
    # Loop over each regression scenario provided in the dictionary
    for name, (dependent, independent_continuous, independent_discrete) in regressions.items():
    
        # Generate synthetic data from model
        data_synthetic = model.generate(data_original, 1)

        # Compute the inidial CIO to gather the variables
        cio_result = compute_CIO(data_original, data_synthetic, dependent, independent_continuous, independent_discrete)

        # Initialize a dict to store the results
        cio_data = {}

        # Fill the dict with the initial round of data
        for _, row in cio_result.iterrows():
            cio_data[row['Variable']] = [row['CIO']]
        
        # Perform CIO calculation for the specified number of folds
        for fold in range(1,folds):
            # Generate synthetic data from model
            data_synthetic = model.generate(data_original, 1)
            
            # Compute CIO for this fold
            cio_result = compute_CIO(data_original, data_synthetic, dependent, independent_continuous, independent_discrete)
            
            # Collect results for averaging later
            for _, row in cio_result.iterrows():
                cio_data[row['Variable']].append(row['CIO'])
        
        # Calculate the average CIO for each variable and prepare the result row
        for variable, cios in cio_data.items():
            avg_cio = sum(cios) / len(cios)
            results.append({'Regression': name, 'Variable': variable, 'CIO': avg_cio})
    
    # Create DataFrame to hold all results
    result_df = pd.DataFrame(results)
    return result_df

# Function to compute the pMSE
def compute_pMSE(data_original_input, data_synthetic_input, discrete_columns):

    data_original = data_original_input.copy()
    data_synthetic = data_synthetic_input.copy()

    # Get the number of features and samples for the hyperparameters of the Random Forest
    num_features = calculate_features(data_original, discrete_columns)
    num_features = round(np.sqrt(num_features))
    n_samples = data_original.shape[0]
    max_depth = round(num_features*np.log(n_samples))

    # Combine the datasets
    data_original['is_synthetic'] = 0
    data_synthetic['is_synthetic'] = 1
    data_combined = pd.concat([data_original, data_synthetic], ignore_index=True)

    # Convert the categorical variales to onehots
    data_combined = pd.get_dummies(data_combined, columns=discrete_columns)

    # Calculate the proportion of synthetic samples
    N = len(data_combined)
    n_s = sum(data_combined['is_synthetic'])
    c = n_s / N

    # Prepare features and target
    y = data_combined['is_synthetic']
    X = data_combined.drop('is_synthetic', axis=1)

    # Fit a classification model
    model = RandomForestClassifier(random_state=42, 
                                   max_features=num_features, 
                                   max_depth=max_depth)
    model.fit(X, y)

    # Predict the propensity scores on the test set
    p_hat = model.predict_proba(X)[:, 1] 

    # Compute pMSE
    pMSE = np.mean((p_hat - c)**2)

    # Normalize pMSE to be between 0 and 1
    pMSE_star = 1 - 4 * pMSE

    return pMSE, pMSE_star