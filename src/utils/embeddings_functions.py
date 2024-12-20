import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Fragments
from mol2vec.features import mol2alt_sentence

def getMolDescriptors(mol, missingVal=None):
    '''
    Calculate the full list of descriptors for a molecule, expect those starting with fr_
    missingVal is used if the descriptor cannot be calculated

    Parameters:
    - mol (RDKit molecule): Molecule to calculate descriptors for
    - missingVal (float): Value to use if a descriptor cannot be calculated

    Returns:
    - dict: Dictionary containing descriptor names and values
    '''
    res = {}
    filtered_desc_list = [(nm, fn) for nm, fn in Descriptors._descList if not nm.startswith('fr_')]
    for nm,fn in filtered_desc_list:
        # some of the descriptor functions can throw errors if they fail, catch those here:
        try:
            val = fn(mol)
        except:
            # Print the error message:
            import traceback
            traceback.print_exc()
            # And set the descriptor value to whatever missingVal is
            val = missingVal
        res[nm] = val
    return res


def get_RDKIT_molecules(df):
    '''
    Get the RDKit molecules for each unique SMILES string in the dataset. This format allows to calculate molecular descriptors.

    Parameters:
    - df (pd.DataFrame): DataFrame containing the SMILES strings

    Returns:
    - list: List of unique SMILES strings
    - pd.Series: Series containing RDKit molecules for each SMILES string
    '''
    smiles = df['Ligand SMILES'].unique()
    smiles_to_molecules = pd.Series([Chem.MolFromSmiles(smiles) for smiles in smiles])
    return smiles, smiles_to_molecules


# Step 1: Calculate molecular descriptors
def calculate_descriptors(molecule):
    '''
    Calculate all RDKIT descriptors for a molecule.
    '''
    
    # Calculate all descriptors
    all_descriptors = getMolDescriptors(molecule)
    
    
    return all_descriptors

def get_mol2vec_descriptors(mol, model): 
    '''
    Get the mol2vec descriptors for a molecule.
    '''
    identifier = mol2alt_sentence(mol, 1)
    embeddings = [model.wv[token] for token in identifier if token in model.wv]
    val = np.mean(embeddings, axis=0)
    return val
     
