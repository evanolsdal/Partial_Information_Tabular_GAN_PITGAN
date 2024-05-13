from collections import namedtuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from rdt.transformers import ClusterBasedNormalizer, OneHotEncoder

"""
This file contains the data transformer object, which takes the data to be generated and prepares it
for proper use by the PITGAN network. This involves two main steps:

    1) Takes the categorical variables and one hot encodes them
    2) Performs the mode specific normalization on the continuous variables

All code here comes directly from the implementation in "Modeling Tabular Data using Conditional GAN" 
from Xu et. al.

"""

SpanInfo = namedtuple('SpanInfo', ['dim', 'activation_fn'])
ColumnTransformInfo = namedtuple(
    'ColumnTransformInfo', [
        'column_name', 'column_type', 'transform', 'output_info', 'output_dimensions'
    ]
)


class DataTransformer(object):
    """Data Transformer.

    Model continuous columns with a BayesianGMM and normalize them to a scalar between [-1, 1]
    and a vector. Discrete columns are encoded using a OneHotEncoder.
    """

    def __init__(self, max_clusters=10, weight_threshold=0.005):
        """Create a data transformer.

        Args:
            max_clusters (int):
                Maximum number of Gaussian distributions in Bayesian GMM.
            weight_threshold (float):
                Weight threshold for a Gaussian distribution to be kept.
        """
        self._max_clusters = max_clusters
        self._weight_threshold = weight_threshold

        # Extra lists to hold the dimensions of the fully transformed outputs
        self.D_list = []
        self.C_list = []

    def _fit_continuous(self, data):
        """Train Bayesian GMM for continuous columns.

        Args:
            data (pd.DataFrame):
                A dataframe containing a column.

        Returns:
            namedtuple:
                A ``ColumnTransformInfo`` object.
        """
        column_name = data.columns[0]
        gm = ClusterBasedNormalizer(
            missing_value_generation='from_column',
            max_clusters=min(len(data), self._max_clusters),
            weight_threshold=self._weight_threshold
        )
        gm.fit(data, column_name)
        num_components = sum(gm.valid_component_indicator)

        self.C_list.append(int(num_components))

        return ColumnTransformInfo(
            column_name=column_name, column_type='continuous', transform=gm,
            output_info=[SpanInfo(1, 'tanh'), SpanInfo(num_components, 'softmax')],
            output_dimensions=1 + num_components)

    def _fit_discrete(self, data):
        """Fit one hot encoder for discrete column.

        Args:
            data (pd.DataFrame):
                A dataframe containing a column.

        Returns:
            namedtuple:
                A ``ColumnTransformInfo`` object.
        """
        column_name = data.columns[0]
        ohe = OneHotEncoder()
        ohe.fit(data, column_name)
        num_categories = len(ohe.dummies)

        self.D_list.append(int(num_categories))

        # Special value to store the number of dimensions

        return ColumnTransformInfo(
            column_name=column_name, column_type='discrete', transform=ohe,
            output_info=[SpanInfo(num_categories, 'softmax')],
            output_dimensions=num_categories)

    def fit(self, raw_data, discrete_columns=()):
        """Fit the ``DataTransformer``.

        Fits a ``ClusterBasedNormalizer`` for continuous columns and a
        ``OneHotEncoder`` for discrete columns.

        This step also counts the #columns in matrix data and span information.
        """
        
        # Ensure that the discrete data is always moved to the front column
        raw_data = place_discrete(raw_data, discrete_columns)
        self.discrete_columns = discrete_columns

        self.output_info_list = []
        self.output_dimensions = 0
        self.dataframe = True

        if not isinstance(raw_data, pd.DataFrame):
            self.dataframe = False
            # work around for RDT issue #328 Fitting with numerical column names fails
            discrete_columns = [str(column) for column in discrete_columns]
            column_names = [str(num) for num in range(raw_data.shape[1])]
            raw_data = pd.DataFrame(raw_data, columns=column_names)

        self._column_raw_dtypes = raw_data.infer_objects().dtypes
        self._column_transform_info_list = []
        for column_name in raw_data.columns:
            if column_name in discrete_columns:
                column_transform_info = self._fit_discrete(raw_data[[column_name]])
            else:
                column_transform_info = self._fit_continuous(raw_data[[column_name]])

            self.output_info_list.append(column_transform_info.output_info)
            self.output_dimensions += column_transform_info.output_dimensions
            self._column_transform_info_list.append(column_transform_info)

    def _transform_continuous(self, column_transform_info, data):
        column_name = data.columns[0]
        flattened_column = data[column_name].to_numpy().flatten()
        data = data.assign(**{column_name: flattened_column})
        gm = column_transform_info.transform
        transformed = gm.transform(data)

        #  Converts the transformed data to the appropriate output format.
        #  The first column (ending in '.normalized') stays the same,
        #  but the lable encoded column (ending in '.component') is one hot encoded.
        output = np.zeros((len(transformed), column_transform_info.output_dimensions))
        output[:, 0] = transformed[f'{column_name}.normalized'].to_numpy()
        index = transformed[f'{column_name}.component'].to_numpy().astype(int)
        output[np.arange(index.size), index + 1] = 1.0

        return output

    def _transform_discrete(self, column_transform_info, data):
        ohe = column_transform_info.transform
        return ohe.transform(data).to_numpy()

    def _synchronous_transform(self, raw_data, column_transform_info_list):
        """Take a Pandas DataFrame and transform columns synchronous.

        Outputs a list with Numpy arrays.
        """
        column_data_list = []
        for column_transform_info in column_transform_info_list:
            column_name = column_transform_info.column_name
            data = raw_data[[column_name]]
            if column_transform_info.column_type == 'continuous':
                column_data_list.append(self._transform_continuous(column_transform_info, data))
            else:
                column_data_list.append(self._transform_discrete(column_transform_info, data))

        return column_data_list

    def _parallel_transform(self, raw_data, column_transform_info_list):
        """Take a Pandas DataFrame and transform columns in parallel.

        Outputs a list with Numpy arrays.
        """
        processes = []
        for column_transform_info in column_transform_info_list:
            column_name = column_transform_info.column_name
            data = raw_data[[column_name]]
            process = None
            if column_transform_info.column_type == 'continuous':
                process = delayed(self._transform_continuous)(column_transform_info, data)
            else:
                process = delayed(self._transform_discrete)(column_transform_info, data)
            processes.append(process)

        return Parallel(n_jobs=-1)(processes)

    def transform(self, raw_data):
        """Take raw data and output a matrix data."""

        # Ensure that the discrete data is always moved to the front column
        raw_data = place_discrete(raw_data, self.discrete_columns)

        if not isinstance(raw_data, pd.DataFrame):
            column_names = [str(num) for num in range(raw_data.shape[1])]
            raw_data = pd.DataFrame(raw_data, columns=column_names)

        # Only use parallelization with larger data sizes.
        # Otherwise, the transformation will be slower.
        if raw_data.shape[0] < 500:
            column_data_list = self._synchronous_transform(
                raw_data,
                self._column_transform_info_list
            )
        else:
            column_data_list = self._parallel_transform(
                raw_data,
                self._column_transform_info_list
            )

        return np.concatenate(column_data_list, axis=1).astype(float)

    def _inverse_transform_continuous(self, column_transform_info, column_data, sigmas, st):
        gm = column_transform_info.transform
        data = pd.DataFrame(column_data[:, :2], columns=list(gm.get_output_sdtypes()))
        data[data.columns[1]] = np.argmax(column_data[:, 1:], axis=1)
        if sigmas is not None:
            selected_normalized_value = np.random.normal(data.iloc[:, 0], sigmas[st])
            data.iloc[:, 0] = selected_normalized_value

        return gm.reverse_transform(data)

    def _inverse_transform_discrete(self, column_transform_info, column_data):
        ohe = column_transform_info.transform
        data = pd.DataFrame(column_data, columns=list(ohe.get_output_sdtypes()))
        return ohe.reverse_transform(data)[column_transform_info.column_name]

    def inverse_transform(self, data, sigmas=None):
        """Take matrix data and output raw data.

        Output uses the same type as input to the transform function.
        Either np array or pd dataframe.
        """
        st = 0
        recovered_column_data_list = []
        column_names = []
        for column_transform_info in self._column_transform_info_list:
            dim = column_transform_info.output_dimensions
            column_data = data[:, st:st + dim]
            if column_transform_info.column_type == 'continuous':
                recovered_column_data = self._inverse_transform_continuous(
                    column_transform_info, column_data, sigmas, st)
            else:
                recovered_column_data = self._inverse_transform_discrete(
                    column_transform_info, column_data)

            recovered_column_data_list.append(recovered_column_data)
            column_names.append(column_transform_info.column_name)
            st += dim

        recovered_data = np.column_stack(recovered_column_data_list)
        recovered_data = (pd.DataFrame(recovered_data, columns=column_names)
                          .astype(self._column_raw_dtypes))

        return recovered_data

    def convert_column_name_value_to_id(self, column_name, value):
        """Get the ids of the given `column_name`."""
        discrete_counter = 0
        column_id = 0
        for column_transform_info in self._column_transform_info_list:
            if column_transform_info.column_name == column_name:
                break
            if column_transform_info.column_type == 'discrete':
                discrete_counter += 1

            column_id += 1

        else:
            raise ValueError(f"The column_name `{column_name}` doesn't exist in the data.")

        ohe = column_transform_info.transform
        data = pd.DataFrame([value], columns=[column_transform_info.column_name])
        one_hot = ohe.transform(data).to_numpy()[0]
        if sum(one_hot) == 0:
            raise ValueError(f"The value `{value}` doesn't exist in the column `{column_name}`.")

        return {
            'discrete_column_id': discrete_counter,
            'column_id': column_id,
            'value_id': np.argmax(one_hot)
        }
    
    # Function which gets the list of discrete categories and continuous categories
    def get_relevant_dimensions(self):

        D_list = self.D_list
        C_list = self.C_list

        return D_list, C_list
    

    # Function which seends the generator output to a representation where the continuous variables are encoded 
    # according to their modes
    def transform_modes(self, data):

        # First convert the data into the transformed representation
        data_transform = self.transform(data)
        data_transform = move_continuous(data_transform, self.D_list, self.C_list)

        # Then take the columns corresponding to modes and covert them to categorical discrete variables
        data_modes = convert_mode_to_categorical(data_transform, self.D_list, self.C_list)

        # Then get the discrete columns
        data_cat = data[self.discrete_columns]

        # Concatenate them 
        output = pd.concat([data_cat, data_modes], axis=1)

        return output

"""
The next part adds a couple of transformations which ensures that the input and output from the 
DataTransformer are in the appropriate form the PITGAN training and inferance
"""

# Function which moves the discrete columns to the from of the dataframe for input to the DataTransformer.
# Note input should be a pandas df
def place_discrete(data, discrete_columns):

    # Ensure that discrete_columns is a list of unique values to avoid potential errors
    discrete_columns = list(dict.fromkeys(discrete_columns))

    # Validate that all discrete_columns exist in the DataFrame
    missing_cols = [col for col in discrete_columns if col not in data.columns]
    if missing_cols:
        raise ValueError(f"The following discrete columns are not in the DataFrame: {missing_cols}")

    # Separate the discrete columns from the continuous ones
    continuous_columns = [col for col in data.columns if col not in discrete_columns]

    # Concatenate the column names, with discrete ones first
    reordered_columns = discrete_columns + continuous_columns

    # Reorder the DataFrame accordingly
    reordered_data = data[reordered_columns]

    return reordered_data


# Function which takes the transformed output from the DataTransformer and reoreders the continuous 
# parts of the transformation output. Note input should be numpy array
def move_continuous(data, D_list, C_list):

    # Calculate the start index of the continuous data
    start_continuous = sum(D_list)
    
    # Initialize a list to hold the indices of continuous columns
    continuous_indices = []
    
    # Initialize a new array to hold the reordered data
    reordered_data = data[:, :start_continuous]  # Start with discrete one-hot columns
    mode_columns = []
    
    # Calculate the indices for the continuous and mode columns
    current_index = start_continuous
    for modes in C_list:
        # The continuous column is the first column for each continuous variable
        continuous_indices.append(current_index)
        # The next 'modes' columns are the mode columns
        mode_columns.extend(list(range(current_index + 1, current_index + 1 + modes)))
        # Update the current index for the next continuous variable
        current_index += 1 + modes
    
    # Append the mode columns to the reordered data
    reordered_data = np.hstack([reordered_data, data[:, mode_columns]])
    
    # Append the continuous columns at the end, maintaining their order
    reordered_data = np.hstack([reordered_data, data[:, continuous_indices]])
    
    return reordered_data


# Function which takes the output from the generator and moves the continuous columns back into the 
# correct order for the inverse mapping of the DataTransformer. Note input should be a tf tensor
def reverte_continuous(data, D_list, C_list):
    
    # Calculate the starting index of continuous data in the original format
    start_continuous = sum(D_list)
    
    # Extract the discrete one-hot columns
    discrete_data = data[:, :start_continuous]
    
    # Extract the continuous columns from the end
    continuous_columns = data[:, -len(C_list):]
    
    # Extract the mode columns which are positioned just after discrete columns and before continuous columns
    mode_columns = data[:, start_continuous:start_continuous + sum(C_list)]
    
    # Initialize a list to store the reorganized columns
    reorganized_data = [discrete_data]
    
    # Current index to track the extraction point for mode columns
    current_index = 0
    for i, modes in enumerate(C_list):
        # Insert the continuous column for the current continuous variable before its mode columns
        continuous_col_for_var = continuous_columns[:, i:i+1]
        reorganized_data.append(continuous_col_for_var)
        
        # Insert mode columns for the current continuous variable
        mode_cols_for_var = mode_columns[:, current_index:current_index + modes]
        reorganized_data.append(mode_cols_for_var)
        
        # Update the current index for mode columns
        current_index += modes
    
    # Concatenate all parts back together
    reorganized_data_np = np.hstack(reorganized_data)
    
    return reorganized_data_np


# Logic for estraxting modes as descrete categorical variables
def convert_mode_to_categorical(output, D_list, C_list):
    # Calculate the start index for slicing output array
    start_idx = sum(D_list)
    
    # Create an empty DataFrame
    df = pd.DataFrame()
    
    # Loop through each category dimension and its corresponding size in C_list
    for k, dim in enumerate(C_list):
        # Calculate the end index for slicing
        end_idx = start_idx + dim
        
        # Slice the output array to get the one-hot encoded columns for this category
        one_hot_data = output[:, start_idx:end_idx]
        
        # Convert one-hot encoded data to categorical indices
        category_indices = np.argmax(one_hot_data, axis=1)
        
        # Create a pandas Categorical column from indices
        category_labels = [f'm_{j+1}' for j in range(dim)]  # Categories as m_1, m_2, ..., m_dim
        categorical_column = pd.Categorical.from_codes(category_indices, categories=category_labels, ordered=True)
        
        # Add the categorical column to the DataFrame
        df[f'C_{k+1}'] = categorical_column
        
        # Update the start index for the next iteration
        start_idx = end_idx
    
    return df

