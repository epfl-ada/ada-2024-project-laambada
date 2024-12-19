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
import os
from gensim.models import Word2Vec

# Processing
from utils.embeddings_functions import get_RDKIT_molecules, calculate_descriptors, get_mol2vec_descriptors

# 1. Chose embeddings
generate_RDKIT_emb = False
generate_Morgan_emb = False
generate_Mol2Vec_emb = True

# 2. Data 
data_folder = 'data'
df_clean = load_data(zip_file_path = f'{data_folder}/clean_subset.csv.zip')
n_batch = len(df_clean)
start_batch = 0
end_batch =  len(df_clean)
df = df_clean[start_batch:end_batch] # df_clean[:n_batch] #Security check

# 3. Molecules
smiles, smiles_to_molecules = get_RDKIT_molecules(df)

### DEBUG
print(len(df), len(smiles), len(smiles_to_molecules))

# 4. RDKIT
if generate_RDKIT_emb:
    descriptor_data = []
    for mol in tqdm(smiles_to_molecules):
        descriptor_data.append(calculate_descriptors(mol))
    save_embeddings(smiles, descriptor_data, name=f'RDKIT_descriptors_{start_batch}_{end_batch}', folder='data')

# 5. Morgan Fingerpint
if generate_Morgan_emb:
    morgan_generator = GetMorganGenerator(radius=2, fpSize=1024)
    ecfp_embeddings = []
    for mol in tqdm(smiles_to_molecules):
        ecfp_embeddings.append(morgan_generator.GetFingerprint(mol))

    ecfp_descriptors_list = [list(fp) for fp in ecfp_embeddings]
    save_embeddings(smiles, ecfp_descriptors_list, name=f'Morgan_Fingerprint_{start_batch}_{end_batch}', folder=data_folder)

# 6. Mol2Vec
if generate_Mol2Vec_emb:
    pretrained_model_path = os.path.join(os.getcwd(), 'models/model_300dim.pkl') 
    model = Word2Vec.load(pretrained_model_path)
    print('number of unique identifiers', len(model.wv.key_to_index))
    mol2vec_embeddings = []
    for mol in tqdm(smiles_to_molecules):
        mol2vec_embeddings.append(get_mol2vec_descriptors(mol, model))
    #mol2vec_descriptors_df = pd.DataFrame(mol2vec_embeddings)
    save_embeddings(smiles, mol2vec_embeddings, name=f'Mol2Vec_{start_batch}_{end_batch}', folder=data_folder)