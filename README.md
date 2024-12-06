# The Ligand That Kinases Your Heart: Investigating the Binding of the Tyrosine Kinase Protein Family to Their Ligands

## Abstract
Drug discovery is, mostly an exercise in binding optimization, crucial for desired drug-target interaction. In this project, we aim to characterize ligands of the Tyrosine Kinase family of proteins. Members of the Tyrosine Kinase family are involved in many biological pathways and pose significant challenges in drug design due to off-target effects. By choosing structurally similar proteins, we get access to a measure of binding specificity, as ligands often bind similar targets with various degrees of affinity.

We will explore the landscape of ligands available for these targets, hunting for features that make these ligands good (or bad) binders. We hope that uncovered features will enable a meaningful interpretation of binding data. Furthermore, we will leverage ML based methods to learn to predict binding affinities from a ligand and thus reconstruct specific 

---

## Research Questions
- How similar/diverse, and specific, are the ligands of the Tyrosine Kinase protein family?
- What structural features in ligands are associated with better binding affinities?
- Will ML based model provide enough information on the kind of structures that make a ligand bind ? Can we leverage it to recreate new ligands ?

---

## Additional Datasets
- PANTHER accessed via UniProt API to classify targets to their respective families

---

## Methods

### Defining What is Affinity
After exploring different affinity metrics, we decided to base our binding affinity analysis on the Ki measurement. In most cases, a drug aims to inhibit its target (or, if we are studying off-target effects, we want to ensure we are not inhibiting the wrong target). The dataset contains a sufficient number of samples for both Ki and IC50. While IC50 depends on assay conditions (e.g., target concentration, temperature) Ki is a more accurate and intrinsic measure, closely related to the molecular structure. This allows for better comparison across experiments and a direct link to the fundamental features of the proteins.

### The Protein Family
Considering the important size of BindingDB, we made a choice to analyze a particular family of proteins. This choice will enable us to study the specificity of interactions between the ligands and their targets. We used the PANTHER classification system, accessed through UniProt API, to classify targets into families. We then discarded families that had less than 5 targets and sorted families by average number of ligands per target. 
After inspection of the remaining possibilities, we decided to concentrate our efforts on the Tyrosine Kinase family.

### Characterizing the Chemical Properties of the Ligands of the Family
To characterize the ligands studied for the family protein Tyrosine Kinase, we investigate their chemical properties with rdkit. Properties like molecular weight, nature of the functional groups, hydrophobicity and electric charge are important molecular features for drug discovery. [The role of ligand efficiency metrics in drug discovery, (Hopkins et al., 2014)].
- **Data Cleaning:** Remove ligands that can not be processed by rdkit.
- **Univariate Data Analysis:** 
  - Investigate the different properties and observe the distributions.
  - Over all ligands.
  - With respect to the target: This will allow us to weigh further analysis. If a target was investigated for way more ligands than the others, this should be detangled from the data.
- **Multivariate Analysis:** Investigate how the different properties of the ligands are related to the Ki.

### To What Extent Are the Ligands of the Same Target Similar?
To answer this question we will represent the ligands in various embedding spaces, starting from their SMILES description. Then we will measure their similarity and study the existence of clusters to see to what extent the ligands corresponding to a same target are more similar compared to the others ligands. 
The used embedding spaces will be: 
- **Morgan Fingerprints (ECFP):** A discrete embedding based on the local structure and generally aligns with chemical substructure similarities.
- **RDKitDescriptors:** A continuous embedding based on the physicochemical properties of the molecules. 
- **Mol2Vec:** A pre-trained ML model (https://github.com/samoturk/mol2vec/tree/master/examples/models) to build a vector representation of the molecules that will enable us to detect more global and subtle differences but is less interpretable than the two previous spaces. 

For each embedding space, we will study the clusters by:
- Applying a dimensionality reduction technique (PCA/ t-SNE) to visualize the points in 2D.
- Creating a heatmap with the similarity between each point (using Tanimoto or Euclidean distance).
- Calculating the silhouette of each target group.
- Performing k-means clustering to double-check the results. 

### Investigating a Protein of Interest
Among the family, we will choose a specific target molecule. The goal of this part is to study in more detail a target of interest and characterize its ligands. We aim to investigate whether we can define, based on the features extracted and their embeddings, what makes a good ligand from the target's perspective. We will then explore what our features and classification mean, and whether we can correlate them to the structure and function of the target of interest. At this stage, we will be able to make inferences about the biological relevance of the target for certain diseases.

---

## ML Inference 

### Feature Selection
We will utilize the ligand properties extracted using RDKit, as well as the embeddings obtained from Morgan Fingerprints, RDKitDescriptors, and mol2vec. We will use feature selection most likely based on feature importance derived from tree-based models, to identify the most significant features contributing to binding affinity.

### Model Training
For binding affinity prediction, we will experiment mainly with explainable predictors including:
- Random Forest Regressor
- Gradient Boosting Regressor 
- SVMs

We reserve the possibility to investigate more expressive models depending on the results achieved with those more basic ones. In particular, small transformer-based, pretrained models trained with a specific head might bring better performances if tree-based methods do not suffice as these methods proved very efficient in recent publications.

---

## Timeline and Group Organization 

### Week 10:
- Get the results from the embedding part (Laura)
- Create a nice visualization of the embedding results (Maud)
- Investigate the target side: What can we say about the ligand of a particular target? (Barbara)
- Make inferences (Alexandre)
- Compare result from inference with the existing ligand structures (Aya)

### Week 11:
- Brainstorming about the website (Everyone)
- Additional information if we want to frame our analysis within the context of a disease (storytelling and data organization) (Everyone)
- Add visualizations (based on our needs after the brainstorming; each of us will work on extracting visualizations for our part)

### Week 12:
- Writing the website and report (Everyone; Alexandre – coordinator)

### Week 13:
- Writing the website and report (Everyone will participate in putting our results on the website, and Alexandre will coordinate everything and make it look nice)

### Week 14:
- Submission
