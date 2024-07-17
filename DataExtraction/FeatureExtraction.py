import logging
import os

import neurokit2 as nk
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class PPGFeatureExtractor:
    def __init__(self, base_dir, output_csv_path, problems_txt_path):
        self.base_dir = base_dir
        self.output_csv_path = output_csv_path
        self.problems_txt_path = problems_txt_path
        self.all_participant_data = []
        self.problematic_participants = []

    def run(self):
        """
        Orchestrates the entire PPG feature extraction process.
        """
        for folder_name in os.listdir(self.base_dir):
            if folder_name.endswith('_GSR'):
                participant_id = folder_name.split('_')[0]
                folder_path = os.path.join(self.base_dir, folder_name)
                csv_file_path = os.path.join(folder_path, 'activity1.csv')

                if os.path.isfile(csv_file_path):
                    analyze_df = self.process_ppg_signals(csv_file_path, participant_id)
                    if analyze_df is not None:
                        self.all_participant_data.append(analyze_df)
                else:
                    logging.warning(f"File not found: {csv_file_path}. Skipping participant {participant_id}...")
                    self.problematic_participants.append(participant_id)

        self.export_features_and_problems()

    def process_ppg_signals(self, csv_file_path, participant_id):
        """
        Process PPG signals for a participant.
        """
        df = self.load_and_clean_data(csv_file_path)
        if df.empty:
            logging.warning(f"No valid data loaded for participant {participant_id}. Skipping...")
            self.problematic_participants.append(participant_id)
            return None

        if 'PPG' not in df.columns or 'time' not in df.columns:
            logging.warning(f"'PPG' or 'time' columns not found in {csv_file_path}. Skipping participant {participant_id}...")
            self.problematic_participants.append(participant_id)
            return None

        try:
            sampling_rate = self.calculate_sampling_rate(df)
            if sampling_rate == -1:
                logging.warning(f"Invalid sampling rate for participant {participant_id}. Skipping...")
                self.problematic_participants.append(participant_id)
                return None

            ppg_cleaned = nk.ppg_clean(df['PPG'].values, sampling_rate=sampling_rate, method='elgendi')
            ppg_signals, info = nk.ppg_process(ppg_cleaned, sampling_rate=sampling_rate)
            analyze_df = nk.ppg_analyze(ppg_signals, sampling_rate=sampling_rate)
            analyze_df['P_Id'] = participant_id
            return analyze_df
        except Exception as e:
            logging.error(f"Error processing PPG signals for participant {participant_id}: {e}")
            self.problematic_participants.append(participant_id)
            return None

    def load_and_clean_data(self, file_path):
        """
        Load and clean PPG data from a CSV file.
        """
        try:
            df = pd.read_csv(file_path)
            df['time'] = pd.to_datetime(df['time'], errors='coerce')
            df['PPG'] = pd.to_numeric(df['PPG'], errors='coerce')
            df = df.dropna()
            return df
        except Exception as e:
            logging.error(f"Error loading and cleaning data from {file_path}: {e}")
            return pd.DataFrame()

    def calculate_sampling_rate(self, ppg_df):
        """
        Calculate the sampling rate from the PPG data.
        """
        if 'time' not in ppg_df.columns:
            return -1

        try:
            ppg_df['time'] = pd.to_datetime(ppg_df['time'])
            time_diffs = ppg_df['time'].diff().dt.total_seconds().dropna()
            median_sampling_interval = time_diffs.median()
            if median_sampling_interval == 0:
                return -1
            sampling_rate = int(1 / median_sampling_interval)
            return sampling_rate
        except Exception as e:
            logging.error(f"Error calculating sampling rate: {e}")
            return -1

    def export_features_and_problems(self):
        """
        Export processed PPG features to CSV and write problematic participant IDs to a text file.
        """
        if not self.all_participant_data:
            logging.warning("No participant data to export.")
            return

        try:
            master_df = pd.concat(self.all_participant_data, ignore_index=True)
            columns_ordered = ['P_Id'] + [col for col in master_df.columns if col != 'P_Id']
            master_df = master_df[columns_ordered]
            master_df = master_df.sort_values(by=['P_Id'])
            master_df.to_csv(self.output_csv_path, index=False)
            logging.info(f"Successfully exported features to {self.output_csv_path}")
        except Exception as e:
            logging.error(f"Error exporting features to {self.output_csv_path}: {e}")

        try:
            with open(self.problems_txt_path, 'w') as f:
                for participant_id in self.problematic_participants:
                    f.write(f"{participant_id}\n")
            logging.info("Problematic participant IDs written to problems.txt")
        except Exception as e:
            logging.error(f"Error writing problematic participant IDs to {self.problems_txt_path}: {e}")


if __name__ == "__main__":
    BASE_DIR = '/home/vivek/Desktop/PPG-Analysis/GSR_data'
    OUTPUT_CSV_PATH = '/home/vivek/DataspellProjects/Project-PPG/Datasets/features.csv'
    PROBLEMS_TXT_PATH = '/home/vivek/DataspellProjects/Project-PPG/Datasets/problems_features.txt'

    pipeline = PPGFeatureExtractor(BASE_DIR, OUTPUT_CSV_PATH, PROBLEMS_TXT_PATH)
    pipeline.run()
