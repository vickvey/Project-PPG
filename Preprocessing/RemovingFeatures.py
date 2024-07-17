"""This script loads data.csv file, counts missing values, removes columns with missing values exceeding a threshold (here 20), and saves the resulting DataFrame to a new CSV file data_cleaned.csv.
"""

import pandas as pd

def load_csv(file_path):
    """Load a CSV file into a pandas DataFrame."""
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return None

def count_missing_values(df):
    """Count missing values in a DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame for which to count missing values
        
    Returns:
        dict: A dictionary containing the count of missing values for each column with at least one missing value
    """
    missing_values = df.isna().sum()
    missing_values = missing_values[missing_values > 0]  # Filter out columns with zero missing values
    return missing_values.to_dict()

def remove_columns_with_missing_values(df, threshold):
    """Remove columns with missing values exceeding the threshold.
    
    Args:
        df (pd.DataFrame): DataFrame from which to remove columns
        threshold (int): Threshold for the number of missing values
        
    Returns:
        pd.DataFrame: DataFrame with specified columns removed
    """
    missing_data_info = count_missing_values(df)
    cols_to_remove = [col for col, count in missing_data_info.items() if count > threshold]
    return df.drop(columns=cols_to_remove)

def save_csv(df, file_path):
    """Save a DataFrame to a CSV file."""
    try:
        df.to_csv(file_path, index=False)
    except Exception as e:
        print(f"Error saving CSV file: {e}")

def main(input_path, output_path, missing_value_threshold):
    """Main function to load, process, and save the DataFrame."""
    data = load_csv(input_path)
    if data is not None:
        data = remove_columns_with_missing_values(data, missing_value_threshold)
        save_csv(data, output_path)

if __name__ == "__main__":
    input_path = 'Datasets/data.csv'
    output_path = 'Datasets/data_cleaned.csv'  # Changed output file name to avoid overwriting the original
    missing_value_threshold = 20
    main(input_path, output_path, missing_value_threshold)
