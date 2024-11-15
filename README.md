# The Ligand That Kinases Your Heart: Investigating the Binding of the Tyrosine Kinase Protein Family to Their Ligands

## Abstract
Drug discovery is mostly an exercise in binding optimization, crucial for desired drug-target interaction. In this project, we aim to characterize ligands of the Tyrosine Kinase family of proteins. Members of the Tyrosine Kinase family are involved in many biological pathways and pose significant challenges in drug design due to off-target effects. By choosing structurally similar proteins, we get access to a measure of binding specificity, as ligands often bind similar targets with various degrees of affinity.

We will explore the landscape of ligands available for these targets, hunting for features that make these ligands good (or bad) binders. We hope that uncovered features will enable a meaningful interpretation of binding data. Furthermore, we will leverage ML-based methods to learn to predict binding affinities from a ligand and thus reconstruct specific interactions.

---

## Research Questions
1. How similar, diverse, and specific are the ligands of the Tyrosine Kinase protein family?
2. What structural features in ligands are associated with better binding affinities?
3. Can we use ML models to predict binding affinities effectively?

---

## Additional Datasets
- **PANTHER** accessed via UniProt API to classify targets into their respective families.

---

## Methods

### Defining Affinity
- **Metric Used:** Ki measurement (intrinsic, accurate, and linked to molecular structure).
- **Rationale:** Ki allows better comparison across experiments and a direct connection to protein-ligand interaction features.

### Protein Family Selection
- Analyzed protein families using **PANTHER classification**.
- Selected families with at least 5 targets.
- Chose **Tyrosine Kinase** family for its relevance and high average number of ligands per target.

### Characterizing Ligands
- **Tools:** rdkit for chemical properties such as molecular weight, functional groups, hydrophobicity, and electric charge.
- **Data Cleaning:** Removed ligands that rdkit could not process.
- **Univariate and Multivariate Analysis:** Explored the relationship between ligand properties and Ki values.

### Ligand Similarity Analysis
- Represented ligands in different **embedding spaces**:
  1. **Morgan Fingerprints (ECFP):** Discrete embeddings based on chemical substructures.
  2. **RDKitDescriptors:** Continuous embeddings based on physicochemical properties.
  3. **Mol2Vec:** Pre-trained ML model for global molecular representation.
- **Techniques:**
  - Dimensionality reduction (PCA/t-SNE) for visualization.
  - Similarity heatmaps (Tanimoto or Euclidean distance).
  - Clustering (silhouette scores, k-means).

### Investigating a Protein of Interest
- Selected a specific target within the Tyrosine Kinase family.
- Characterized ligands based on extracted features and embeddings.
- Explored correlations between ligand features and the target's structure and function.

---

## ML Inference

### Feature Selection
- Focused on ligand properties and embeddings (e.g., RDKitDescriptors, Morgan Fingerprints, mol2vec).
- Tree-based models for feature importance.

### Model Training
- Predictors tested:
  - Random Forest Regressor
  - Gradient Boosting Regressor
  - SVMs
- Future exploration:
  - Transformer-based models with specialized heads if simpler methods underperform.

---

## Timeline and Group Organization

### Week 10
- Embedding results (Laura).
- Embedding visualizations (Maud).
- Target investigation (Barbara).
- Inferences (Alexandre).
- Compare inferences with ligand structures (Aya).

### Week 11
- Brainstorm website content (Everyone).
- Add disease context (Everyone).
- Extract visualizations for respective parts (Everyone).

### Week 12
- Website and report writing (Everyone; Alexandre – coordinator).

### Week 13
- Finalize website and report (Everyone).

### Week 14
- Submission.

---

## References
- Hopkins et al. (2014), *The Role of Ligand Efficiency Metrics in Drug Discovery*.
