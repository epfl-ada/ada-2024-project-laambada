# 1. Libraries
import numpy as np
import time
import pandas as pd
from tqdm import tqdm

from scripts.load_and_save import load_data, save_embeddings

# Ligand analysis 
from rdkit import Chem
from rdkit.Chem import Descriptors, Fragments
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

# Processing
from utils.embeddings_functions import get_RDKIT_molecules, calculate_descriptors

# 2. Data 
data_folder = 'data'
df_clean = load_data(zip_file_path = f'{data_folder}/clean_subset.csv.zip')
n_batch = len(df_clean)
start_batch = 0
end_batch = 40000
df = df_clean[start_batch:end_batch] # df_clean[:n_batch] #Security check

# 3. Molecules
smiles, smiles_to_molecules = get_RDKIT_molecules(df)

### DEBUG
print(len(df), len(smiles), len(smiles_to_molecules))

# 4. RDKIT
descriptor_data = []
for mol in tqdm(smiles_to_molecules):
    descriptor_data.append(calculate_descriptors(mol))
save_embeddings(smiles, descriptor_data, name=f'RDKIT_descriptors_{start_batch}_{end_batch}', folder='data')

# 5. Morgan Fingerpint
morgan_generator = GetMorganGenerator(radius=2, fpSize=1024)
ecfp_embeddings = []
for mol in tqdm(smiles_to_molecules):
    ecfp_embeddings.append(morgan_generator.GetFingerprint(mol))

ecfp_descriptors_list = [list(fp) for fp in ecfp_embeddings]
save_embeddings(smiles, ecfp_descriptors_list, name=f'Morgan_Fingerprint_{start_batch}_{end_batch}', folder=data_folder)
