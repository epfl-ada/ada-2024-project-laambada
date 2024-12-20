# The Target That Kinases Your Heart: Investigating the Binding of the Tyrosine Kinase Protein Family to Their Ligands

* [Data story](https://epfl-ada.github.io/ada-2024-project-laambada/)
* [Cloud with plots](https://drive.google.com/drive/folders/1I1yEa0xd_tJtfgWgKuvr7xtn3u0SQR-C)

## Abstract
Our project explores binding preferences of the TYROSINE-PROTEIN KINASE HOPSCOTCH family of proteins (referred to as the tyrosine kinase family in this project). Members of this family are involved in many biological [pathways](https://pubmed.ncbi.nlm.nih.gov/33430292/) and pose significant [challenges](https://genesdev.cshlp.org/content/17/24/2998.full.pdf) in drug design due to off-target effects.

We propose to views the binding of tyrosine kinases to their ligands through the prism of embeddings. In this project, we start by selecting adequate binding metrics and providing a chemical characterization of the ligands that bind to the targets of our interest. We then construct embedings using [RDKit](https://www.rdkit.org/) Descriptors, [Mol2vec]((https://pubs.acs.org/doi/10.1021/acs.jcim.7b00616)), Morgan fingerprints, and a concatenation of all embeddings. To evaluate the relevance of resulting embeddings, we analyze clustering generated upon dimensionality reduction of embedded ligands.  

Finally, we construct a multilayer perceptron and a graph neural network to infer binding affinities from the structures of ligands.  

## Research Questions
- What does the chemical makeup of the ligands of the tyrosine kinase protein family look like?
- What chemical characteristics of ligands are associated with enhanced binding?
- Is there an embedding space that captures the affinity and specificity of target-ligand interactions? 
- Can we predict binding affinities from ligand structures?

## Additional Datasets
- PANTHER accessed via UniProt API to classify targets to their respective families

## Methods

### The Protein Family
Considering the important size of BindingDB, we made a choice to analyze a particular family of proteins. We used the PANTHER classification system, accessed through UniProt API, to classify targets into families. We then discarded families that had less than 5 targets and sorted families by average number of ligands per target. After inspection of the remaining possibilities, we decided to concentrate our efforts on the tyrosine kinase family.

### Defining Efficient Binding
After exploring different affinity metrics, we decided to base our binding affinity analysis on the Ki and IC50 measurements. In most cases, a drug aims to inhibit its target, therefore inhibition-associated measurements seem like a right fit. The dataset contains a sufficient number of samples for both Ki and IC50. While IC50 depends on assay conditions (e.g., target concentration, temperature), Ki is a more accurate and intrinsic measure, closely related to the molecular structure.

### Characterizing the Chemical Properties of the Ligands of the Family
To characterize the ligands of tyrosine kinases, we investigate their chemical properties with RDKit. Properties like molecular weight, nature of the functional groups, hydrophobicity, and electric charge are [important molecular features](https://pubmed.ncbi.nlm.nih.gov/24481311/) for drug discovery. After initial preprocessing, we conduct:

- Univariate Data Analysis: 
  - Investigate different chemical properties and observe their distributions.
- Multivariate Analysis: Investigate how the different properties of the ligands are related to Ki and IC50.

### To What Extent Are the Ligands of the Same Target Similar?
To answer this question, we represent the ligands in various embedding spaces, starting from their SMILES representation. Then we apply dimensionality reduction and study the existence of clusters to see to what extent the ligands corresponding to a same target are clustered. In addition, we study whether there are any trends in Ki and IC50 distribution of embedded ligands.  

The used embedding spaces are: 

- **Morgan Fingerprints (ECFP):** A discrete embedding based on the local structure that generally aligns with chemical substructure similarities.
- **RDKit Descriptors:** A continuous embedding capturing the physicochemical properties of the molecules that we construct from the entierty of RDKit Descriptors. 
- **Mol2Vec:** A [pre-trained ML model](https://github.com/samoturk/mol2vec/tree/master/examples/models) to build a vector representation of the molecules that allows us to detect more global but also subtle differences.
- **Concatenated embeddings**: A concatenation of the three previous embeddings with the hope to bring the best of the three embeddings together. 

For each embedding space, we apply PCA, followed by UMAP, to study whether and how ligands cluster together.

### ML Inference 

**Multilayer perceptron (MLP):** We train a MLP with four linear layers and ReLU activation using concatenated embeddings of ligands and ordinally encoded targets as input, and IC50 measurments as output.

**Graph neural network (GNN):** Inspired by what is considered to be state of the art in terms of ML models in drug discovery [[1](https://pubs.acs.org/doi/10.1021/acs.jcim.1c00975), [2](https://www.cell.com/cell/fulltext/S0092-8674(20)30102-1), [3](https://www.nature.com/articles/s41586-023-06887-8)], we train our own graph neural network (GNN) to predict pIC50. Each ligand is first represented as a graph containing the information about its structure. We add some relevant RDKit Descriptors to the nodes and some bond information to the edges of the graphs. Then we train a GNN that takes as input a graph of a ligand and its ordinally encoded target, and outputs pIC50.    

## Group Organization

- Aygul: family selection; website redaction
- Maud: chemical characterization; data visualization 
- Laura: embedding spaces; data visualization 
- Barbara: Metric investigation; website redaction
- Alexandre: ML Inference; embedding generation
