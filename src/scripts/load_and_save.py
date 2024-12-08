import zipfile
import pandas as pd

# Load the whole dataset here as a data frame
def load_full_dataset(zip_file_path = 'data/BindingDB_All_tsv.zip', tsv_file_name = 'BindingDB_All.tsv'):
    # Open the zip file
    with zipfile.ZipFile(zip_file_path, 'r') as z:
        # Open the TSV file within the zip
        with z.open(tsv_file_name) as file:
            # Read the TSV file into a DataFrame
            df = pd.read_csv(file, sep='\t', on_bad_lines='skip')

    return df

def load_data(zip_file_path = 'src/data/clean_subset.csv.zip'):
    df = pd.read_csv(zip_file_path, compression='zip')
    return df
