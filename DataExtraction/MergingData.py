import pandas as pd

def main(features_path, labels_path, output_path, key_col='P_Id'):
    """Load, merge, and save DataFrames."""
    features_df = pd.read_csv(features_path)
    labels_df = pd.read_csv(labels_path)
    merged_data = pd.merge(features_df, labels_df, on=key_col)
    merged_data.to_csv(output_path, index=False)

if __name__ == "__main__":
    features_path = 'Datasets/features.csv'
    labels_path = 'Datasets/labels.csv'
    output_path = 'Datasets/merged.csv'
    
    main(features_path, labels_path, output_path)
