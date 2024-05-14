import pandas as pd
import numpy as np

"""
This file stores the privacy measure
"""

# This function computes the Targeted Correct Attribution Probability measure of a given dataset
def compute_TCAP(data_real, data_synth, list_of_keys, transformer, WEAP_threshold):

    # First transform data to the mode representation of continuous variables
    data_real = transformer.transform_modes(data_real)
    data_synth = transformer.transform_modes(data_synth)

    # Initialize a list to store results for each target across all keys
    global_results_list = []

    # Loop over each set of keys
    for keys in list_of_keys:
        # Get all columns that are not part of the keys (these are potential targets)
        potential_targets = [col for col in data_synth.columns if col not in keys]

        data_synth = data_synth.copy()

        # Initialize a list to store results for each target
        results_list = []

        # Loop over each potential target column
        for target in potential_targets:

            # Skip processing if the target column has only one unique value in either dataset
            if len(data_real[target].unique()) <= 1:
                continue

            # Calculate WEAP for each record in the synthetic dataset
            data_synth['WEAP'] = data_synth.apply(
                lambda row: (
                    ((data_synth[target] == row[target]) & (data_synth[keys] == row[keys]).all(axis=1)).sum() /
                    (data_synth[keys] == row[keys]).all(axis=1).sum()
                    if (data_synth[keys] == row[keys]).all(axis=1).sum() > 0 else 0
                ), axis=1
            )

            # Filter for records with a WEAP score at least equal to WEAP_threshold
            susceptible_records = data_synth[data_synth['WEAP'] >= WEAP_threshold].copy()

            if susceptible_records.empty:
                average_tcap = 0
            else:
                # Calculate TCAP for susceptible records using real dataset
                susceptible_records['TCAP'] = susceptible_records.apply(
                    lambda row: (
                        ((data_real[target] == row[target]) & (data_real[keys] == row[keys]).all(axis=1)).sum() /
                        (data_real[keys] == row[keys]).all(axis=1).sum()
                        if (data_real[keys] == row[keys]).all(axis=1).sum() > 0 else 0
                    ), axis=1
                )

                # Calculate the average TCAP for this target
                average_tcap = susceptible_records['TCAP'].mean()

            # Append the result for this target to the list
            results_list.append({'Variable': target, 'TCAP': average_tcap})

        # Append results to the global results list with an additional key count column
        for result in results_list:
            result.update({'Num_Keys': f"{len(keys)}-keys"})
            global_results_list.append(result)

    # Convert list of global results to a DataFrame
    results_df = pd.DataFrame(global_results_list)

    return results_df



def compute_TCAP_new(data_real, data_synth, list_of_keys, transformer, WEAP_threshold):
    # Transform data to the mode representation of continuous variables
    data_real = transformer.transform_modes(data_real)
    data_synth = transformer.transform_modes(data_synth)

    global_results_list = []

    # Look over each set of keys
    for keys in list_of_keys:
        # Identify potential targets
        potential_targets = [col for col in data_synth.columns if col not in keys]

        for target in potential_targets:
            # Group by keys and target and calculate counts
            synth_group = data_synth.groupby(keys + [target]).size().reset_index(name='count_synth')
            real_group = data_real.groupby(keys + [target]).size().reset_index(name='count_real')

            # Calculate WEAP by merging synthetic groups on keys
            synth_keys_group = data_synth.groupby(keys).size().reset_index(name='total_synth')
            weap_data = pd.merge(synth_group, synth_keys_group, on=keys)
            weap_data['WEAP'] = weap_data['count_synth'] / weap_data['total_synth']

            # Filter groups based on WEAP threshold
            susceptible_records = weap_data[weap_data['WEAP'] >= WEAP_threshold]

            if susceptible_records.empty:
                average_tcap = 0
            else:
                # Calculate TCAP by applying weap calculation but to the real dataset
                real_keys_group = data_real.groupby(keys).size().reset_index(name='total_real')
                tcap_data = pd.merge(real_group, real_keys_group, on=keys)
                tcap_data['TCAP'] = tcap_data['count_real'] / tcap_data['total_real']

                # Merge these TCAP scores to the susceptible records according to the keys and targets
                tcap_merged = pd.merge(susceptible_records, tcap_data, on=keys + [target], how='left')
                tcap_merged['TCAP'] = tcap_merged['TCAP'].fillna(0)  
                tcap_merged['TCAP_full'] = tcap_merged['TCAP'] * tcap_merged['count_synth']

                # Calculate the average TCAP for this target
                total_susceptible = tcap_merged['count_synth'].sum()
                average_tcap = tcap_merged['TCAP_full'].sum() / total_susceptible 

            # Store results
            global_results_list.append({
                'Variable': target, 
                'TCAP': average_tcap, 
                'Num_Keys': f"{len(keys)}-keys"
            })

    # Convert list of results to a DataFrame
    results_df = pd.DataFrame(global_results_list)
    return results_df