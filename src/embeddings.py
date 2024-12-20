'''
File name: embeddings.py
Author: Alexandre Sallinen, Maud Dupont-Roc, Laura Gambaretto
Date created: 20 November 2024
Date last modified: 15 December 2024
Python Version: 3.7
'''

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


def generate_molecule_embeddings(
    smiles_data,
    generate_RDKIT=False,
    generate_Morgan=False, 
    generate_Mol2Vec=False,
    generate_All=False,
    data_folder='data',
    mol2vec_model_path='models/model_300dim.pkl'
):
    """
    Generate molecular embeddings for a list of SMILES strings.
    
    Args:
        smiles_data (pd.DataFrame): DataFrame containing SMILES strings
        generate_RDKIT (bool): Generate RDKIT descriptors
        generate_Morgan (bool): Generate Morgan fingerprints 
        generate_Mol2Vec (bool): Generate Mol2Vec embeddings
        data_folder (str): Folder to save embeddings
        mol2vec_model_path (str): Path to pretrained Mol2Vec model
        
    Returns:
        dict: Dictionary containing generated embeddings
    """
    # Get RDKIT molecules
    smiles, smiles_to_molecules = get_RDKIT_molecules(smiles_data)
    embeddings = {}
    
    start_batch = 0
    end_batch = len(smiles_data)
    
    # Generate RDKIT descriptors
    if generate_RDKIT:
        descriptor_data = []
        for mol in tqdm(smiles_to_molecules):
            descriptor_data.append(calculate_descriptors(mol))
        save_embeddings(smiles, descriptor_data, 
                       name=f'RDKIT_descriptors_{start_batch}_{end_batch}', 
                       folder=data_folder)
        embeddings['rdkit'] = descriptor_data

    # Generate Morgan fingerprints
    if generate_Morgan:
        morgan_generator = GetMorganGenerator(radius=2, fpSize=1024)
        ecfp_embeddings = []
        for mol in tqdm(smiles_to_molecules):
            ecfp_embeddings.append(morgan_generator.GetFingerprint(mol))
        ecfp_descriptors_list = [list(fp) for fp in ecfp_embeddings]
        save_embeddings(smiles, ecfp_descriptors_list, 
                       name=f'Morgan_Fingerprint_{start_batch}_{end_batch}', 
                       folder=data_folder)
        embeddings['morgan'] = ecfp_descriptors_list

    # Generate Mol2Vec embeddings
    if generate_Mol2Vec:
        model = Word2Vec.load(mol2vec_model_path)
        mol2vec_embeddings = []
        for mol in tqdm(smiles_to_molecules):
            mol2vec_embeddings.append(get_mol2vec_descriptors(mol, model))
        save_embeddings(smiles, mol2vec_embeddings, 
                       name=f'Mol2Vec_{start_batch}_{end_batch}', 
                       folder=data_folder)
        embeddings['mol2vec'] = mol2vec_embeddings
    
    # Generate all embeddings that have already been computed (flatten)
    if generate_All:
        if 'rdkit' not in embeddings or 'morgan' not in embeddings or 'mol2vec' not in embeddings:
            raise ValueError('RDKIT, Morgan and Mol2Vec embeddings must be generated first.')
        
        all_embeddings = []
        for mol in tqdm(smiles_to_molecules):
            all_embeddings.append(np.concatenate([
                embeddings['rdkit'][smiles_to_molecules.index(mol)],
                embeddings['morgan'][smiles_to_molecules.index(mol)],
                embeddings['mol2vec'][smiles_to_molecules.index(mol)]
            ]))
        
        save_embeddings(smiles, all_embeddings, 
                       name=f'All_{start_batch}_{end_batch}', 
                       folder=data_folder)
        embeddings['all'] = all_embeddings
        
    return embeddings

if __name__ == '__main__':
    # Load data
    data = load_data('data/ligands.csv')
    
    # Generate molecular embeddings
    embeddings = generate_molecule_embeddings(data, generate_RDKIT=True, generate_Morgan=True, generate_Mol2Vec=True, generate_All=True)