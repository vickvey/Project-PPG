import os
import numpy as np
import pandas as pd
import neurokit2 as nk

class PPGFeatureExtractor:
    def __init__(self, base_dir, output_csv_path, problems_txt_path):
        # Class variables
        self.base_dir = base_dir
        self.output_csv_path = output_csv_path
        self.problems_txt_path = problems_txt_path
        self.all_participant_data = []
        self.problematic_participants = []

    def load_and_clean_data(self, file_path):
        """
        Load data from a CSV file and clean it by converting time to datetime and PPG to numeric.

        Parameters:
        file_path (str): Path to the CSV file to be loaded.

        Returns:
        pd.DataFrame: Cleaned DataFrame with time and PPG columns.
        """
        df = pd.read_csv(file_path)
        df['time'] = pd.to_datetime(df['time'], errors='coerce')
        df['PPG'] = pd.to_numeric(df['PPG'], errors='coerce')
        df = df.dropna()
        return df


    def calculate_sampling_rate(self, ppg_df):
        """
        Calculate the sampling rate from the PPG DataFrame.
        """
        if 'PPG' not in ppg_df.columns or 'time' not in ppg_df.columns:
            return -1

        try:
            ppg_df['time'] = pd.to_datetime(ppg_df['time'])
            min_time = ppg_df['time'].min()
            max_time = ppg_df['time'].max()
            total_duration_seconds = (max_time - min_time).total_seconds()

            if total_duration_seconds == 0:
                return -1

            total_data_rows = len(ppg_df['PPG'])
            sampling_rate = int(total_data_rows / total_duration_seconds)
            return sampling_rate
        
        except Exception as e:
            print(f"Error calculating sampling rate: {e}")
            return -1

    def process_ppg_signals(self, csv_file_path, P_Id):
        """
        Process PPG signals for a single participant.
        """
        df = self.load_and_clean_data(csv_file_path)
        if 'PPG' in df.columns and 'time' in df.columns:
            ppg_signal_series = df['PPG']
            sampling_rate = self.calculate_sampling_rate(df)

            if sampling_rate == -1:
                print(f"Invalid sampling rate for participant {P_Id}. Skipping...")
                self.problematic_participants.append(P_Id)
                return None

            ppg_cleaned = nk.ppg_clean(ppg_signal_series.values, sampling_rate=sampling_rate, method='elgendi')
            ppg_signals, info = nk.ppg_process(ppg_cleaned, sampling_rate=sampling_rate)
            analyze_df = nk.ppg_analyze(ppg_signals, sampling_rate=sampling_rate)
            analyze_df['P_Id'] = P_Id
            return analyze_df
        else:
            print(f"Warning: 'PPG' or 'time' columns not found in {csv_file_path}. Skipping participant {P_Id}...")
            self.problematic_participants.append(P_Id)
            return None

    def export_features_and_problems(self):
        """
        Export the extracted features and the list of problematic participants to CSV and TXT files respectively.
        """
        if self.all_participant_data:
            master_df = pd.concat(self.all_participant_data, ignore_index=True)
            columns_ordered = ['P_Id'] + [col for col in master_df.columns if col != 'P_Id']
            master_df = master_df[columns_ordered]
            master_df = master_df.sort_values(by=['P_Id'])
            master_df.to_csv(self.output_csv_path, index=False)
            print(f"Successfully exported features to {self.output_csv_path}")

            if self.problematic_participants:
                with open(self.problems_txt_path, 'w') as f:
                    for participant_id in self.problematic_participants:
                        f.write(f"{participant_id}\n")
                print("Problematic participant IDs written to problems.txt")
        else:
            print("No participant data found or processed.")

    def run(self):
        """
        Execute the pipeline to process PPG signals and extract features.
        """
        for folder_name in os.listdir(self.base_dir):
            if folder_name.endswith('_GSR'):
                P_Id = folder_name.split('_')[0]
                folder_path = os.path.join(self.base_dir, folder_name)
                csv_file_path = os.path.join(folder_path, 'activity1.csv')

                if os.path.isfile(csv_file_path):
                    analyze_df = self.process_ppg_signals(csv_file_path, P_Id)
                    if analyze_df is not None:
                        self.all_participant_data.append(analyze_df)
                else:
                    print(f"File not found: {csv_file_path}. Skipping participant {P_Id}...")
                    self.problematic_participants.append(P_Id)

        self.export_features_and_problems()

if __name__ == "__main__":
    BASE_DIR = '/home/vivek/Desktop/PPG-Analysis/GSR_data'
    OUTPUT_CSV_PATH = '/home/vivek/DataspellProjects/Project-PPG/features.csv'
    PROBLEMS_TXT_PATH = '/home/vivek/DataspellProjects/Project-PPG/problems_features.txt'

    pipeline = PPGFeatureExtractor(BASE_DIR, OUTPUT_CSV_PATH, PROBLEMS_TXT_PATH)
    pipeline.run()
