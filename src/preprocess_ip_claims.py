import numpy as np
import pandas as pd
import pickle as pl
import os
import glob
from concurrent.futures import ThreadPoolExecutor
import time

def reduce_mem_usage(df):
    """ Iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024**2
    # print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    # print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    # print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    return df

def read_csv_in_chunks(file_path, chunksize=10000):
    print(f"Reading file in chunks: {file_path}")
    chunks = []
    for chunk in pd.read_csv(file_path, chunksize=chunksize):
        chunks.append(chunk)
    combined_df = pd.concat(chunks, ignore_index=True)
    return reduce_mem_usage(combined_df)

def concatenate_claims(directories):
    combined_ip_claims = pd.DataFrame()

    patterns = {
        "Inpatient_Claims": "*Inpatient_Claims*.csv",
    }

    with ThreadPoolExecutor() as executor:
        future_to_df = {}
        for claim_type, pattern in patterns.items():
            for directory in directories:
                # print(f"Searching for files in directory: {directory} with pattern: {pattern}")
                for file_path in glob.glob(os.path.join(directory, pattern)):
                    # print(f"Found file: {file_path} for claim type: {claim_type}")
                    future = executor.submit(read_csv_in_chunks, file_path)
                    future_to_df[future] = claim_type

        for future in future_to_df:
            result = future.result()
            if future_to_df[future] == "Inpatient_Claims":
                combined_ip_claims = pd.concat([combined_ip_claims, result], ignore_index=True)

    return combined_ip_claims

class ClaimsProcessor:
    def __init__(self, claims_file):
        start_time = time.time()
        print("Loading claims data...")
        self.claims = pd.read_pickle(claims_file)
        self.claims['ICD9_PRCDR_CD_1'] = self.claims['ICD9_PRCDR_CD_1'].apply(lambda x: str(int(x)) if pd.notnull(x) else x)
        print(f"Loaded claims data in {time.time() - start_time:.2f} seconds.")
        print('-'*50)

    @staticmethod
    def calculate_ip_readmissions(df):
        df['CLM_ADMSN_DT'] = pd.to_datetime(df['CLM_ADMSN_DT'], format='%Y%m%d', errors='coerce')
        df['NCH_BENE_DSCHRG_DT'] = pd.to_datetime(df['NCH_BENE_DSCHRG_DT'], format='%Y%m%d', errors='coerce')

        df.sort_values(by=['DESYNPUF_ID', 'CLM_ADMSN_DT'], inplace=True)

        df['PREV_DSCHRG_DT'] = df.groupby('DESYNPUF_ID')['NCH_BENE_DSCHRG_DT'].shift(1)
        df['DAYS_SINCE_LAST_DISCHARGE'] = (df['CLM_ADMSN_DT'] - df['PREV_DSCHRG_DT']).dt.days

        df['IP_READMIT_30DAYS'] = df['DAYS_SINCE_LAST_DISCHARGE'].apply(lambda x: 1 if x <= 30 else 0)

        return df

    def preprocess(self):
        start_time = time.time()
        print("Calculating inpatient readmissions...")
        self.claims = self.calculate_ip_readmissions(self.claims)
        print(f"Calculated inpatient readmissions in {time.time() - start_time:.2f} seconds.")
        print('-'*50)

        start_time = time.time()
        print("Counting unique patients...")
        self.unique_patients = self.claims['DESYNPUF_ID'].nunique()
        print(f"Counted unique patients in {time.time() - start_time:.2f} seconds.")
        print('-'*50)

        start_time = time.time()
        print("Counting unique claims per patient...")
        unique_claims_counts = self.claims.groupby('DESYNPUF_ID')['CLM_ID'].nunique()
        print(f"Counted unique claims per patient in {time.time() - start_time:.2f} seconds.")
        print('-'*50)

        start_time = time.time()
        print("Filtering patients with 3 or more visits...")
        patients_with_3_or_more_visits = unique_claims_counts[unique_claims_counts >= 3].index
        self.total_patients = len(patients_with_3_or_more_visits)
        self.filtered_claims = self.claims[self.claims['DESYNPUF_ID'].isin(patients_with_3_or_more_visits)]
        print(f"Filtered patients with 3 or more visits in {time.time() - start_time:.2f} seconds.")
        print('-'*50)

        start_time = time.time()
        print("Counting readmissions...")
        readmission_counts = self.filtered_claims.groupby('DESYNPUF_ID')['IP_READMIT_30DAYS'].sum()
        self.patients_with_readmissions = readmission_counts[readmission_counts > 0]
        self.number_of_patients_with_readmissions = len(self.patients_with_readmissions)
        self.patients_without_readmissions = readmission_counts[readmission_counts == 0]
        self.number_of_patients_with_no_readmissions = len(self.patients_without_readmissions)
        print(f"Counted readmissions in {time.time() - start_time:.2f} seconds.")
        print('-'*50)

    def report(self):
        print("Generating report...")
        print("Number of patients with 3 or more visits and at least one 30 day re-admission:", self.number_of_patients_with_readmissions)
        print("Number of patients with 3 or more visits and no 30 day re-admission:", self.number_of_patients_with_no_readmissions)
        print(f'Total number of patients with 3 or more visits: {self.total_patients} out of {self.unique_patients} total patients ({100 * self.total_patients / self.unique_patients:.0f}%)')
        print('-'*50)

    def get_codes(self):
        start_time = time.time()
        print("Extracting codes...")
        data = self.filtered_claims.copy()

        columns = ['DESYNPUF_ID', 'CLM_ID', 'ADMTNG_ICD9_DGNS_CD', 'ICD9_DGNS_CD_1', 'ICD9_DGNS_CD_2', 'ICD9_DGNS_CD_3',
                   'ICD9_DGNS_CD_4', 'ICD9_DGNS_CD_5', 'ICD9_DGNS_CD_6', 'ICD9_DGNS_CD_7', 'ICD9_DGNS_CD_8', 'ICD9_DGNS_CD_9', 'ICD9_DGNS_CD_10',
                   'CLM_DRG_CD', 'IP_READMIT_30DAYS', 'ICD9_PRCDR_CD_1', 'ICD9_PRCDR_CD_2', 'ICD9_PRCDR_CD_3', 'ICD9_PRCDR_CD_4']
        subset_data = data[columns]

        # Collecting all ICD9 codes into a single column array for each row
        subset_data['ALL_ICD9_CODES'] = subset_data[['ADMTNG_ICD9_DGNS_CD', 'ICD9_DGNS_CD_1', 'ICD9_DGNS_CD_2',
                                                     'ICD9_DGNS_CD_3', 'ICD9_DGNS_CD_4', 'ICD9_DGNS_CD_5']].values.tolist()

        # Exploding the list into rows, dropping NaNs, and dropping duplicates per patient
        subset_data = subset_data.explode('ALL_ICD9_CODES').dropna(subset=['ALL_ICD9_CODES']).drop_duplicates(subset=['DESYNPUF_ID', 'CLM_ID', 'ALL_ICD9_CODES'])

        # Grouping by DESYNPUF_ID and CLM_ID and collecting unique ICD9 codes as a list
        icd9_codes = subset_data.groupby(by=['DESYNPUF_ID', 'CLM_ID'])['ALL_ICD9_CODES'].apply(lambda x: list(x.unique())).reset_index()
        icd9_codes = icd9_codes.rename(columns={'ALL_ICD9_CODES': 'ICD9_CODE'})

        # Continue with PROC Codes in the same manner
        subset_data['ALL_PROC_CODES'] = subset_data[['ICD9_PRCDR_CD_1', 'ICD9_PRCDR_CD_2', 'ICD9_PRCDR_CD_3', 'ICD9_PRCDR_CD_4']].values.tolist()
        subset_data = subset_data.explode('ALL_PROC_CODES').dropna(subset=['ALL_PROC_CODES']).drop_duplicates(subset=['DESYNPUF_ID', 'CLM_ID', 'ALL_PROC_CODES'])

        # Grouping by DESYNPUF_ID and CLM_ID and collecting unique PROC codes as a list
        proc_codes = subset_data.groupby(by=['DESYNPUF_ID', 'CLM_ID'])['ALL_PROC_CODES'].apply(lambda x: list(x.unique())).reset_index()
        proc_codes = proc_codes.rename(columns={'ALL_PROC_CODES': 'ICD9_PROC_CODES'})

        # Merging all results
        result = pd.merge(icd9_codes, proc_codes, on=['DESYNPUF_ID', 'CLM_ID'], how='inner')
        result = result.rename(columns={'DESYNPUF_ID': 'SUBJECT_ID', 'CLM_ID': 'HADM_ID'})

        print(f"Extracted codes in {time.time() - start_time:.2f} seconds.")
        print('-'*50)
        return result, icd9_codes, proc_codes

    def save_unique_codes(self, icd9_codes, proc_codes, output_dir='./vocab'):
        start_time = time.time()
        print("Saving unique ICD and PROC codes...")

        # Ensure the output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Get all unique ICD codes, converting each to string and filtering out NaN values
        unique_icd_codes = set()
        for codes in icd9_codes['ICD9_CODE']:
            for code in codes:
                if pd.notna(code):
                    unique_icd_codes.add(str(code))

        # Save unique ICD codes to dx-vocab.txt file
        with open(os.path.join(output_dir, 'dx-vocab.txt'), 'w') as file:
            file.write('\n'.join(unique_icd_codes))

        print('Created a vocab with {} unique ICD codes.'.format(len(unique_icd_codes)))

        # Get all unique PROC codes, converting each to string and filtering out NaN values
        unique_proc_codes = set()
        for codes in proc_codes['ICD9_PROC_CODES']:
            for code in codes:
                if pd.notna(code):
                    unique_proc_codes.add(str(code))

        # Save unique PROC codes to proc-vocab.txt file
        with open(os.path.join(output_dir, 'proc-vocab.txt'), 'w') as file:
            file.write('\n'.join(unique_proc_codes))

        print('Created a vocab with {} unique PROC codes.'.format(len(unique_proc_codes)))
        print(f"Saved unique codes in {time.time() - start_time:.2f} seconds.")
        print('-'*50)

if __name__ == "__main__":
    directories = [f'./cms_de10_data/sample_{i}' for i in range(1, 21)]
    combined_ip_claims = concatenate_claims(directories)
    print('-'*50)
    print(f'There are {combined_ip_claims.shape[0]} inpatient claims in the dataset with {combined_ip_claims.shape[1]} columns.')

    # Output the concatenated dataframes to pickle
    with open('./data/combined_ip_claims.pkl', 'wb') as file:
        pl.dump(combined_ip_claims, file)

    print('-'*50)
    print('Processing the combined IP claims...')
    print('-'*50)
    
    # Process the combined IP claims
    processor = ClaimsProcessor('./data/combined_ip_claims.pkl')
    processor.preprocess()
    processor.report()
    result, icd9_codes, proc_codes = processor.get_codes()
    processor.save_unique_codes(icd9_codes, proc_codes)
    print(result.head())

    os.makedirs('./data', exist_ok=True)

    with open('./data/data-comb-visit.pkl', 'wb') as file:
        pl.dump(result, file)
