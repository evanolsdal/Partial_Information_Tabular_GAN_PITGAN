import numpy as np
import pandas as pd

"""
This defines functions to create basic synthetic datasets which can then be used for testing 
"""

"""
    Generate synthetic data with arbitrary categories and associated multivariate distributions.
    
    Parameters:
    - cats: List of initial weights for categories, to be normalized to form a PMF.
    - mu: List of means for the multivariate Gaussian distributions for each category.
    - sig: List of covariance matrices for the multivariate Gaussian distributions for each category.
    - N: Number of samples to generate.
    
    Returns:
    - A numpy array of generated samples with each row containing a one-hot encoded category
      followed by a 2D sample drawn from the corresponding multivariate Gaussian distribution.
    """
def generate_basic_2D(cats, cat_names, mu, sig, N):

    # Make sure all the inputs have the correct dimensions
    if not (len(cats) == len(mu) == len(sig) == len(cat_names)):
        raise ValueError("cats, mu, sig, and cat_names must all be of the same length.")
    
    # Normalize category weights to create a PMF
    total_weight = sum(cats)
    pmf = [cat / total_weight for cat in cats]
    
    # Draw N samples based on the PMF
    categories = np.random.choice(len(cats), p=pmf, size=N)
    
    # Initialize lists for categorical names and 2D Gaussian samples
    categorical_data = [cat_names[cat] for cat in categories]
    gaussian_samples = np.array([np.random.multivariate_normal(mu[cat], sig[cat]) for cat in categories])
    
    # Create a DataFrame
    df = pd.DataFrame(gaussian_samples, columns=['Dimension_1', 'Dimension_2'])
    df['Category'] = categorical_data
    
    return df




