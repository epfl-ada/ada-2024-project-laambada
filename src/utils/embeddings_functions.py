import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Fragments
from mol2vec.features import mol2alt_sentence

def getMolDescriptors(mol, missingVal=None):
    ''' Calculate the full list of descriptors for a molecule, expect those starting with fr_
    
        missingVal is used if the descriptor cannot be calculated
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
    smiles = df['Ligand SMILES'].unique()
    smiles_to_molecules = pd.Series([Chem.MolFromSmiles(smiles) for smiles in smiles])
    #mask = smiles_to_molecules.isna()
    #smiles = smiles[mask]
    #smiles_to_molecules.dropna(inplace=True)
 
    return smiles, smiles_to_molecules


# Step 1: Calculate molecular descriptors
def calculate_descriptors(molecule):
    """Calculate key molecular descriptors for a molecule"""
    
    # Calculate all descriptors
    all_descriptors = getMolDescriptors(molecule)
    
    
    return all_descriptors

def get_mol2vec_descriptors(mol, model): 
    """ Get the mol2vec descriptors for a molecule """
    identifier = mol2alt_sentence(mol, 1)
    embeddings = [model.wv[token] for token in identifier if token in model.wv]
    val = np.mean(embeddings, axis=0)
    return val
     
