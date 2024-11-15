'''
File name: extract_subdataset.py
Author: Aygul Bayramova, Maud Dupont-Roc
Date created: 7 November 2024
Date last modified: 15 November 2024
'''

import pandas as pd
import zipfile

def extract_subdataset(data_path: str, tsv_file_name, sample_size: int, seed: int):
    '''
    Extract a sample of the dataset from a zip file.
    '''
    with open(data_path, 'r') as f:
        with zipfile.ZipFile(f, 'r') as z:
            with z.open(tsv_file_name) as file:
                df = pd.read_csv(file, sep='\t', on_bad_lines='skip')
    sample_df = df.sample(n=sample_size, random_state=seed)
    return sample_df