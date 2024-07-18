import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def save_corr_matrix_to_excel(df: pd.DataFrame, excel_path: str) -> pd.DataFrame:
    """
    Calculates the correlation matrix of the DataFrame and saves it to an Excel file.

    Parameters:
    - df: pandas DataFrame
      The original DataFrame with features.
    - excel_path: str
      The file path to save the correlation matrix as an Excel file.

    Returns:
    - correlation_matrix: pandas DataFrame
      The calculated correlation matrix.
    """
    # Calculate the correlation matrix
    correlation_matrix = df.corr()

    # Save the correlation matrix to an Excel file
    correlation_matrix.to_excel(excel_path)
    print(f'Correlation matrix is exported to: {excel_path}')
    return correlation_matrix

def save_corr_heatmap(correlation_matrix: pd.DataFrame, heatmap_path: str):
    """
    Generates a correlation heatmap and saves it as an image file.

    Parameters:
    - correlation_matrix: pandas DataFrame
      The correlation matrix of the dataset.
    - heatmap_path: str
      The file path to save the heatmap image.
    """
    # Generate a correlation heatmap
    plt.figure(figsize=(20, 20))
    sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', fmt='.2f')

    # Save the heatmap as an image file
    plt.savefig(heatmap_path)
    print(f'Correlation heatmap is saved to: {heatmap_path}')

def get_high_correlation_pairs(correlation_matrix: pd.DataFrame, threshold: float = 0.80) -> pd.DataFrame:
    """
    Returns a DataFrame of feature pairs with correlation values
    either greater than or equal to the threshold or less than or equal to -threshold.

    Parameters:
    - correlation_matrix: pandas DataFrame
      The correlation matrix of the dataset.
    - threshold: float (default: 0.80)
      The threshold for considering a correlation as high.

    Returns:
    - high_corr_df: pandas DataFrame
      A DataFrame containing the feature pairs and their correlation values.
    """
    # Create an empty list to store the feature pairs and their correlation values
    high_correlation_pairs = []

    # Iterate through the correlation matrix
    for i in range(len(correlation_matrix.columns)):
        for j in range(i + 1, len(correlation_matrix.columns)):
            # Get the correlation value
            corr_value = correlation_matrix.iloc[i, j]

            # Check if the correlation value meets the criteria
            if corr_value >= threshold or corr_value <= -threshold:
                # Add the feature pair and correlation value to the list as a tuple
                feature_pair = (correlation_matrix.columns[i], correlation_matrix.columns[j], corr_value)
                high_correlation_pairs.append(feature_pair)

    # Create a DataFrame from the list of high correlation pairs
    high_corr_df = pd.DataFrame(high_correlation_pairs, columns=['Feature A', 'Feature B', 'Correlation'])

    # Sort the DataFrame by the absolute value of the correlation in descending order
    high_corr_df['Absolute Correlation'] = high_corr_df['Correlation'].abs()
    high_corr_df = high_corr_df.sort_values(by='Absolute Correlation', ascending=False).drop(columns='Absolute Correlation')

    return high_corr_df

def reduce_high_correlation_features(df: pd.DataFrame, high_corr_df: pd.DataFrame) -> pd.DataFrame:
    """
    Reduces features in the DataFrame `df` based on high correlation pairs in `high_corr_df`.

    Parameters:
    - df: pandas DataFrame
      The original DataFrame with features.
    - high_corr_df: pandas DataFrame
      A DataFrame containing highly correlated feature pairs.

    Returns:
    - reduced_df: pandas DataFrame
      The DataFrame with reduced features.
    """
    # Set to keep track of features to drop
    features_to_drop = set()

    # Iterate over the high correlation pairs
    for index, row in high_corr_df.iterrows():
        feature_a = row['Feature A']
        feature_b = row['Feature B']

        # If both features are not marked to drop, mark one of them
        if feature_a not in features_to_drop and feature_b not in features_to_drop:
            # Mark feature_b to drop (you can change this logic as needed)
            features_to_drop.add(feature_b)

    # Create a list of remaining features
    remaining_features = [feature for feature in df.columns if feature not in features_to_drop]

    # Create the reduced DataFrame with remaining features
    reduced_df = df[remaining_features]

    return reduced_df

def main():
    # Load the dataset
    df = pd.read_csv('Datasets/reduced_I.csv')

    # Store the target `anxiety_meter` column
    labels = df['anxiety_meter']
    df.drop(columns=['anxiety_meter'], inplace=True)

    # File paths for saving the correlation matrix and heatmap
    excel_path = 'DataPreprocessing/Correlation/corr_mat_I.xlsx'
    heatmap_path = 'DataPreprocessing/Correlation/corr_hmap_I.png'

    # Save the correlation matrix to an Excel file and get the correlation matrix
    correlation_matrix = save_corr_matrix_to_excel(df, excel_path)

    # Save the correlation heatmap to a PNG file
    save_corr_heatmap(correlation_matrix, heatmap_path)

    # Get the high correlation pairs
    high_corr_df = get_high_correlation_pairs(correlation_matrix, threshold=0.80)

    # Reduce the features in the dataset
    df_reduced = reduce_high_correlation_features(df, high_corr_df)

    # Create a new DataFrame for the reduced dataset
    df_reduced_final = df_reduced.copy()

    # Add the column `anxiety_meter` back to the reduced dataset
    df_reduced_final['anxiety_meter'] = labels

    # Save the reduced dataset to a CSV file
    df_reduced_final.to_csv('Datasets/reduced_II.csv', index=False)
    print("Reduced dataset is saved to 'Datasets/reduced_II.csv'")

    # Save the correlation matrix of reduced final DataFrame to Excel
    excel_path_reduced = 'DataPreprocessing/Correlation/corr_mat_II.xlsx'
    save_corr_matrix_to_excel(df_reduced_final, excel_path_reduced)

    # Save the correlation heatmap of reduced final DataFrame to PNG
    heatmap_path_reduced = 'DataPreprocessing/Correlation/corr_hmap_II.png'
    correlation_matrix_reduced = df_reduced_final.corr()
    save_corr_heatmap(correlation_matrix_reduced, heatmap_path_reduced)

if __name__ == "__main__":
    main()
