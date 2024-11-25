import zipfile
import pandas as pd

# Load the whole dataset here as a data frame
def load_data(zip_file_path = 'data/BindingDB_All_tsv.zip', tsv_file_name = 'BindingDB_All.tsv'):
    # Open the zip file
    with zipfile.ZipFile(zip_file_path, 'r') as z:
        # Open the TSV file within the zip
        with z.open(tsv_file_name) as file:
            # Read the TSV file into a DataFrame
            df = pd.read_csv(file, sep='\t', on_bad_lines='skip')

    return df

def load_data_direct(tsv_file_name = 'data/BindingDB_All.tsv', affinity = 'Ki (nM)'):
    with open(tsv_file_name) as file:
            # Read the TSV file into a DataFrame
            df = pd.read_csv(file, sep='\t', on_bad_lines='skip', usecols=["Ligand SMILES",
                                                                            'UniProt (SwissProt) Entry Name of Target Chain',
                                                                            affinity,
                                                                           ])

    # Rename the columns to ligand / target / affinity
    df['ligand'] = df['Ligand SMILES']
    df['target'] = df['UniProt (SwissProt) Entry Name of Target Chain']
    df['affinity'] = df[affinity]
    
    # Drop the old columns
    df = df.drop(columns=['Ligand SMILES', 'UniProt (SwissProt) Entry Name of Target Chain', affinity])
    
    return df