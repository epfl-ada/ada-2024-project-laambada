import zipfile
import pandas as pd
import numpy as np

# Load the whole dataset here as a data frame
def load_full_dataset(zip_file_path = '../data/BindingDB_All_tsv.zip', tsv_file_name = 'BindingDB_All.tsv'):
    # Open the zip file
    with zipfile.ZipFile(zip_file_path, 'r') as z:
        # Open the TSV file within the zip
        with z.open(tsv_file_name) as file:
            # Read the TSV file into a DataFrame
            df = pd.read_csv(file, sep='\t', on_bad_lines='skip')

    return df

def load_data(zip_file_path = 'src/data/clean_subset.csv.zip'):
    df = pd.read_csv(zip_file_path, compression='zip')
    try:
        df = log_data(df)
    except:
        pass
    return df

def log_data(df_clean):
    df_clean['pIC50'] = np.where(
    df_clean['IC50 (nM)'] > 0,  # Only apply log10 to positive values
    -np.log10(df_clean['IC50 (nM)'] * 1e-9),  # Transform to molar and take -log10
    np.nan  # Assign NaN for zero or negative values
    )
    df_clean['pKi'] = np.where(
        df_clean['Ki (nM)'] > 0,  # Only apply log10 to positive values
        -np.log10(df_clean['Ki (nM)'] * 1e-9),  # Transform to molar and take -log10
        np.nan  # Assign NaN for zero or negative values
    )

    return df_clean

def save_embeddings(smiles, embeddings, name, folder):
    df_embeddings = pd.DataFrame(embeddings)
    df_embeddings['Ligand SMILES'] = smiles
    df_embeddings.set_index('Ligand SMILES', inplace=True)
    df_embeddings.to_csv(f'{folder}/embeddings_{name}.csv.zip', 
                                  compression='zip',
                                  encoding='utf-8',
                                  index=True)