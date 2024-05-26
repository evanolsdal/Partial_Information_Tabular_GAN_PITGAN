import pandas as pd
import numpy as np

"""
This file stores the privacy measure
"""

# This function computes the Targeted Correct Attribution Probability measure of a given dataset
def compute_TCAP(data_real, data_synth, list_of_keys, transformer, WEAP_threshold):
    # Transform data to the mode representation of continuous variables
    data_real = transformer.transform_modes(data_real)
    data_synth = transformer.transform_modes(data_synth)

    global_results_list = []

    # Look over each set of keys
    for keys in list_of_keys:

        # Compute the totals for this set of keys
        synth_keys_group = data_synth.groupby(keys).size().reset_index(name='total_synth')
        real_keys_group = data_real.groupby(keys).size().reset_index(name='total_real')

        # Identify potential targets
        potential_targets = [col for col in data_synth.columns if col not in keys]

        for target in potential_targets:
            # Group by keys and target and calculate counts
            synth_group = data_synth.groupby(keys + [target]).size().reset_index(name='count_synth')
            real_group = data_real.groupby(keys + [target]).size().reset_index(name='count_real')

            # Calculate WEAP by merging synthetic groups on keys
            weap_data = pd.merge(synth_group, synth_keys_group, on=keys)
            weap_data['WEAP'] = weap_data['count_synth'] / weap_data['total_synth']

            # Filter groups based on WEAP threshold
            susceptible_records = weap_data[weap_data['WEAP'] >= WEAP_threshold]

            if susceptible_records.empty:
                average_tcap = 0
                full_tcap = 0
                average_tcap_real = 0
                full_tcap_real = 0
            else:
                # Calculate TCAP by applying weap calculation but to the real dataset
                tcap_data = pd.merge(real_group, real_keys_group, on=keys)
                tcap_data['TCAP'] = tcap_data['count_real'] / tcap_data['total_real']

                # Merge these TCAP scores to the susceptible records according to the keys and targets
                tcap_merged = pd.merge(susceptible_records, tcap_data, on=keys + [target], how='left')
                tcap_merged['TCAP'] = tcap_merged['TCAP'].fillna(0)  
                tcap_merged['TCAP_full'] = tcap_merged['TCAP'] * tcap_merged['count_synth']
                tcap_merged['TCAP_full_real'] = tcap_merged['TCAP']*tcap_merged['count_real']


                # Calculate the average TCAP for this target
                total_susceptible = tcap_merged['count_synth'].sum()
                average_tcap = tcap_merged['TCAP_full'].sum() / total_susceptible 
                full_tcap = tcap_merged['TCAP_full'].sum()

                total_susceptible_real = tcap_merged['count_real'].sum()                
                if total_susceptible_real == 0:
                    average_tcap_real = 0
                    full_tcap_real = 0
                else:
                    full_tcap_real = tcap_merged['TCAP_full_real'].sum()
                    average_tcap_real = full_tcap_real/total_susceptible_real
                    

            # Store results
            global_results_list.append({
                'Num_Keys': f"{len(keys)}-keys",
                'Variable': target, 
                'TCAP': average_tcap, 
                'TCAP_raw': full_tcap,
                'TCAP_real': average_tcap_real,
                'TCAP_raw_real': full_tcap_real
            })

    # Convert list of results to a DataFrame
    results_df = pd.DataFrame(global_results_list)
    return results_df