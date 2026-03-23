import pandas as pd

"""
This module contains functions for one-hot encoding categorical variables in a DataFrame.
The language  variable is handled separately to ensure that the resulting columns are properly named and formatted
Because the variables are separated by commas.
The function encode data performs one-hot encoding on all categorical variables which are got a single value.

"""

def encode_language(df):
    """
    This function takes a DataFrame as input and performs one-hot encoding on the 'lenguajes_de_programacion_o_tecnologias_que_utilices_en_tu_puesto_actual' column.
    It splits the values in this column by commas, creates new columns for each unique language or technology, and concatenates them back to the original DataFrame.
    Finally, it drops the original column and returns the modified DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame containing the data to be encoded.
    
    """
    lenguajes = df['lenguajes_de_programacion_o_tecnologias_que_utilices_en_tu_puesto_actual'].str.get_dummies(sep=',')
    lenguajes.columns = [col.strip().replace('_', '') for col in lenguajes.columns]
    df = pd.concat([df, lenguajes], axis=1)
    df = df.drop('lenguajes_de_programacion_o_tecnologias_que_utilices_en_tu_puesto_actual', axis=1)
    return df

def encode_data(df):
    """
    
    This function takes a DataFrame as input and performs one-hot encoding on the categorical variables
    It also ensures that any boolean columns are converted to integers (0 and 1).
    The resulting DataFrame is returned, ready for use in machine learning models.

    Args:
        df (pd.DataFrame): The input DataFrame containing the data to be encoded.
    
    """
    df_for_split=pd.get_dummies(df)
    df_for_split = df_for_split.loc[:, ~df_for_split.columns.duplicated()]
    for col in df_for_split.columns:
        if df_for_split[col].dtype == 'bool':
            df_for_split[col] = df_for_split[col].astype(int)
    return df_for_split

