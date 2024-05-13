import numpy as np
import pandas as pd

"""
Some helper functions for the evaluation procecure
"""

# Function that creates a grid of parameters

# Function that helps calcualte the appropriate number of features of dummied dataset for 
# use in the pMSE random forest
def calculate_features(data, discrete_columns):

    # Initialize total feature count with the number of continuous features
    continuous_features = data.shape[1] - len(discrete_columns)
    adjusted_feature_count = continuous_features
    
    # Calculate contribution of each discrete feature
    for column in discrete_columns:
        if column in data.columns:
            num_categories = data[column].nunique()
            adjusted_feature_count += round(np.sqrt(num_categories))
        else:
            raise ValueError(f"Column '{column}' is not present in the dataframe.")

    return adjusted_feature_count