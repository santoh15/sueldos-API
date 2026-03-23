'''
Acá se cargan distintas funciones para el análisis exploratorio de datos (EDA) y la preparación de los datos para el modelado.
Estas funciones incluyen visualizaciones, transformaciones de datos, manejo de valores faltantes, etc.
El objetivo es entender mejor los datos, buscar datos faltantes y prepararlos adecuadamente para el entrenamiento del modelo.

'''
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def clean_column_names(df):
    """
    Standardizes column names by converting to lowercase and replacing spaces with underscores.
    Also applies the same transformation to categorical columns.

    Args:
        df (pd.DataFrame): Input DataFrame to clean.

    Returns:
        pd.DataFrame: DataFrame with cleaned column names.
    """
    df.columns = df.columns.str.lower().str.replace(' ', '_')
    categorical_columns = list(df.dtypes[df.dtypes == 'object'].index)
    for c in categorical_columns:
        df[c] = df[c].str.lower().str.replace(' ', '_')
    return df


def check_nulls(df, name="DataFrame"):
    """
    Prints the count of null values for each column in the DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame to check for nulls.
        name (str): Label to identify the DataFrame in output.
    """

    print(f"\n amount of nulls per column {name}:")
    print(df.isnull().sum())


def df_clean_salary_null(df, relevant_columns):
    """
    Prepares the DataFrame for modeling by selecting relevant columns and dropping rows with null values in the target variable.
    
    Args:
        df (pd.DataFrame): Cleaned DataFrame with necessary columns.
        relevant_columns (list): List of column names to keep for modeling.
        """
    df_model = df[relevant_columns].copy()
    df_model = df_model.dropna(subset=["_sal"])
    return df_model


def count_rows_with_any_null(df, name="DataFrame"):
    """
    Prints how many rows have at least one null value and shows them.

    Args:
        df (pd.DataFrame): Input DataFrame.
        name (str): Label to identify the DataFrame in output.
    """

    null_rows = df[df.isnull().any(axis=1)]
    print(f"\n amount of rows that have at least a NaN value: {len(null_rows)}")
    print(null_rows)


def tipe_of_columns(df):
    """
    Prints the data types of each column in the DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame to check data types.
    """

    print("\n data types of columns:")
    print(df.dtypes)


def plot_distributions(df):
    """
    Plots histograms for  Years of Experience, age and salary.

    Args:
        df (pd.DataFrame): DataFrame that includes those columns.
    """

    fig, axes = plt.subplots(1, 6, figsize=(18, 5))

    sns.histplot(df["_sal"], bins=50, kde=True, ax=axes[0])
    axes[0].set_title("Salary")

    sns.histplot(df["tengo_edad"], bins=20, kde=True, ax=axes[1])
    axes[1].set_title("Age")

    sns.histplot(df["anos_de_experiencia"], bins=20, kde=True, ax=axes[2])
    axes[2].set_title("Years of Experience")

    sns.histplot(df["antiguedad_en_la_empresa_actual"], bins=20, kde=True, ax=axes[3])
    axes[3].set_title("Seniority in Current Company")

    sns.histplot(df["anos_en_el_puesto_actual"], bins=20, kde=True, ax=axes[4])
    axes[4].set_title("Years in Current Position")

    sns.histplot(df["cuantas_personas_tenes_a_cargo"], bins=20, kde=True, ax=axes[5])
    axes[5].set_title("Number of People in Charge")

    plt.tight_layout()
    plt.show()


def count_job_titles(df, threshold):
    """
    Displays the count of job titles that appear more than a given threshold.

    Args:
        df (pd.DataFrame): DataFrame with a 'Job Title' column.
        threshold (int): Minimum number of appearances to be considered.
    """

    job_counts = df["trabajo_de"].value_counts()
    titles_above_N = job_counts[job_counts > threshold]
    total_rows_above_N = titles_above_N.sum()

    print(f"\n Amount of rows with job titles that appear more than {threshold} times: {total_rows_above_N}")
    print(f" Job Titles with more than {threshold} repetitions:\n{titles_above_N}")
    

def count_dedication(df):
    """
    Show the count of dedication levels that appear more than a given threshold.
    Args:
        df (pd.DataFrame): DataFrame with a 'dedicacion' column.

    """

    dedication_counts = df["dedicacion"].value_counts() 

    print(f"\n Amount of rows with dedication levels:")
    print(dedication_counts)


def count_career(df, threshold):
    """
    Displays the count of career that appear more than a given threshold.

    Args:
        df (pd.DataFrame): DataFrame with a 'carrera' column.
        threshold (int): Minimum number of appearances to be considered.
    """

    career_counts = df["carrera"].value_counts()
    careers_above_N = career_counts[career_counts > threshold]
    total_rows_above_N = careers_above_N.sum()

    print(f"\n Amount of rows with career that appear more than {threshold} times: {total_rows_above_N}")
    print(f" Career with more than {threshold} repetitions:\n{careers_above_N}")


def count_seniority(df):
    """
    Show the count of seniority levels that appear more than a given threshold.
    Args:
        df (pd.DataFrame): DataFrame with a 'seniority' column.

    """

    seniority_counts = df["seniority"].value_counts() 

    print(f"\n Amount of rows with seniority levels: ")
    print(seniority_counts)


def column_5_max_min_list(df, column_name):
    """
    Prints the 5 highest and 5 lowest values in the specified column.
    
    Args:
        df (pd.DataFrame): Cleaned DataFrame with the specified column.
        column_name (str): Name of the column to analyze.
    """
    max_values = df[column_name].nlargest(5)
    min_values = df[column_name].nsmallest(5)
    print(f"5 Highest {column_name.replace('_', ' ').title()}:")
    print(max_values)
    print(f"\n5 Lowest {column_name.replace('_', ' ').title()}:")
    print(min_values)


def salary_max_min_list(df):
    """
    Prints the 5 highest and 5 lowest salaries in the dataset.
    
    Args:
        df (pd.DataFrame): Cleaned DataFrame with '_sal'.
    """
    max_salaries = df["_sal"].nlargest(5)
    min_salaries = df["_sal"].nsmallest(5)
    print("5 Highest Salaries:")
    print(max_salaries)
    print("\n5 Lowest Salaries:")
    print(min_salaries)


def remove_salary_outliers(df, column_name='_sal', lower_percentile=0.05, upper_percentile=0.95):
    """
    Removes outliers from the salary column based on specified percentiles.
    This function calculates the lower and upper bounds for the salary based on the given percentiles and
    filters the DataFrame to include only rows where the salary is within these bounds.
    Args:
        df (pd.DataFrame): Input DataFrame containing the salary column.
        column_name (str): Name of the salary column to clean. Default is '_sal'.
        lower_percentile (float): Lower percentile to determine the lower bound. Default is 0.05 (5%).
        upper_percentile (float): Upper percentile to determine the upper bound. Default is 0.95 (95%).
    """
    lower_bound = df[column_name].quantile(lower_percentile)
    upper_bound = df[column_name].quantile(upper_percentile)
    
    print("-" * 30)
    print(f"Clean  Outliers in '{column_name}':")
    print(f"Lower bound ({lower_percentile*100}%): ${lower_bound:,.2f}")
    print(f"Upper bound ({upper_percentile*100}%): ${upper_bound:,.2f}")
    df_cleaned = df[(df[column_name] >= lower_bound) & (df[column_name] <= upper_bound)].copy()
    
    remove_rows = len(df) - len(df_cleaned)
    print(f"Rows removed: {remove_rows} ({(remove_rows/len(df))*100:.2f}% of the dataset)")
    print("-" * 30)
    
    return df_cleaned


def remove_outliers_percentil(df, columna, percentil=0.95):
    """
    Remove outliers from a specified column based on a given percentile.
    This function calculates the upper limit for the specified column based on the given percentile and
    filters the DataFrame to include only rows where the column value is less than or equal to this limit.
    
    Args:   df (pd.DataFrame): Input DataFrame containing the column to clean.
            columna (str): Name of the column from which to remove outliers.
            percentil (float): Percentile to determine the upper limit for outliers. Default is 0.95
    
    Returns:   pd.DataFrame: DataFrame with outliers removed based on the specified percentile.
    
    """
    upper_limit = df[columna].quantile(percentil)
    print(f"Columna '{columna}': Eliminando valores mayores a {upper_limit:.2f}")
    df_clean = df[df[columna] <= upper_limit]
    
    return df_clean


def convert_dolarized_salary_to_int(df):
    """
    Converts the 'sueldo_dolarizado' column to integers by removing any non-numeric characters and converting the result to an integer type.
    
    Args:
        df (pd.DataFrame): Input DataFrame containing the 'sueldo_dolarizado' column.
    Returns:
        pd.DataFrame: DataFrame with the 'sueldo_dolarizado' column converted to integers.
    """
    df.sueldo_dolarizado = df.sueldo_dolarizado.astype(int)
    return df

def convert_salary_actualization_to_binary(df):
    """
    Converts the 'recibis_algun_tipo_de_bono' and 'estas_buscando_trabajo' columns to binary values (0 and 1).
    For 'recibis_algun_tipo_de_bono', 'no' is converted to 0 and any other value is converted to 1.
    
    Args:
        df (pd.DataFrame): Input DataFrame containing the relevant columns.
    Returns:
        pd.DataFrame: DataFrame with the specified columns converted to binary values.
    """
    df['tuviste_actualizaciones_de_tus_ingresos_laborales_durante_el_ultimo_semestre'] = np.where(df['tuviste_actualizaciones_de_tus_ingresos_laborales_durante_el_ultimo_semestre'] == 'no', 0, 1)
    return df


