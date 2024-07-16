import pandas as pd

class LabelGenerator:
    def __init__(self, excel_file, missing_ids):
        self.df = pd.read_excel(excel_file)
        self.missing_ids = missing_ids
        
    def apply_mapping(self, column_name, mapping):
        self.df[column_name] = self.df[column_name].map(mapping)
        
    def add_anxiety_meter_column(self, columns_to_sum):
        self.df['anxiety_meter'] = self.df[columns_to_sum].sum(axis=1)
        
    def remove_missing_ids(self, id_column):
        self.df = self.df[~self.df[id_column].isin(self.missing_ids)]
        
    def save_result_csv(self, csv_file_path, columns_to_save):
        result_df = self.df[columns_to_save]
        result_df.to_csv(csv_file_path, index=False)

def main():
    excel_file = 'labels/participant_response.xlsx'
    
    # Question 2 is reversed, so we need to map the values to the correct ones
    mapping = {1: 5, 2: 4, 3: 3, 4: 2, 5: 1, None: None}
    
    columns_to_sum = ['AS1_1', 'AS1_2', 'AS1_3', 'AS1_4', 'AS1_5']
    
    # These are the ids that are missing from the excel file `participant_response.xlsx`
    # Missing ids have been found from the `participant_info.xlsx` file
    missing_ids = [110, 131, 136, 174, 182, 185, 197, 206, 177]
    
    # The result will be saved in `labels.csv` file
    csv_file_path = 'labels.csv'
    
    label_gen = LabelGenerator(excel_file, missing_ids)
    label_gen.apply_mapping('AS1_2', mapping)
    label_gen.add_anxiety_meter_column(columns_to_sum)
    label_gen.remove_missing_ids('P_Id')
    label_gen.save_result_csv(csv_file_path, ['P_Id', 'anxiety_meter'])

if __name__ == '__main__':
    main()
