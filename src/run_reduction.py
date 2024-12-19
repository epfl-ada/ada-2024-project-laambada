from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import StandardScaler
from sklearn.manifold import TSNE
import numpy as np
import pandas as pd
import os
import plotly.express as px
from umap import UMAP
from src.scripts.load_and_save import load_data
from src.utils.embeddings_plots import create_properties_plot, create_target_plot

def run_pca_reduction(scaled_descriptors, save_path, smiles_data, sensitivity=0.95, ):

    # Apply PCA
    pca = PCA(n_components=sensitivity)
    pca_result = pca.fit_transform(scaled_descriptors)

    print(f"Explained variance ratio: { sum(pca.explained_variance_ratio_):.2f}")
    pca_df = pd.DataFrame(pca_result, columns=[f'PC{i+1}' for i in range(pca_result.shape[1])])
    pca_df.insert(0, 'Ligand SMILES', smiles_data)
    pca_df.set_index('Ligand SMILES', inplace=True)

    pca_df.to_csv(save_path, index=False, compression='zip')
    
    return pca_df, pca    
    
def run_umap_reduction(data, smiles_data, output_path, random_state=42):
    """
    Perform t-SNE dimensionality reduction on the input data.
    
    Args:
        data (np.ndarray): Input data for t-SNE reduction
        smiles_data (pd.Series): SMILES strings corresponding to the data
        output_dir (str): Directory to save the output
        random_state (int): Random seed for reproducibility
        
    Returns:
        pd.DataFrame: DataFrame containing t-SNE results and SMILES
    """
    
    umap = UMAP(verbose=False,
                low_memory=True, )
    umap_result = umap.fit_transform(data)
    # Create output DataFrame
    umap_df = pd.DataFrame(umap_result, columns=['UMAP1', 'UMAP2'])
    umap_df.insert(0, 'Ligand SMILES', smiles_data)
    
    umap_df.set_index('Ligand SMILES', inplace=True)
    # Save results
    umap_df.to_csv(output_path, index=False, compression='zip')
    
    return umap_df

def create_pca_feature_importance_plot(data, pca, fig_output_path, n_embeddings=None, n_components=None):
    # Get feature names from original data
    feature_names = data.columns.tolist()

    # Calculate feature importance (loadings)
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)

    # Create initial DataFrame with loadings
    loadings_df = pd.DataFrame(
        loadings,
        columns=[f'PC{i+1} ({var:.1%})' for i, var in enumerate(pca.explained_variance_ratio_)],
        index=feature_names
    )


    if n_components:
        # only keep top n_components PCs
        loadings_df = loadings_df.iloc[:, :n_components]
        
    if n_embeddings:
        # Get top n_embeddings for each PC based on absolute values
        top_features = []
        for col in loadings_df.columns:
            top_for_pc = loadings_df[col].abs().nlargest(n_embeddings).index.tolist()
            top_features.extend(top_for_pc)
        
        # Keep unique features
        top_features = list(dict.fromkeys(top_features))
        loadings_df = loadings_df.loc[top_features]

    # Create heatmap using Plotly
    fig = px.imshow(
        loadings_df,
        aspect='auto',
        color_continuous_scale='peach',
        title='PCA Feature Importance Heatmap',
        labels={'x': 'Principal Components', 'y': 'Features', 'color': 'Loading'}
    )

    # Update layout
    fig.update_layout(
        width=1000,
        height=800,
        xaxis_side='bottom',
        yaxis={'tickmode': 'linear'}
    )
    
    fig.show()
    fig.write_html(fig_output_path)
        
    return fig

def vizualize_reduction(pca, umap, df, fig_output_path):
    
    df_merged_pca = df.merge(pca, on='Ligand SMILES')
    df_merged_umap = df.merge(umap, on='Ligand SMILES')
    
    fig_target_pca = create_target_plot(df_merged_pca, 'PCA')
    fig_target_umap = create_target_plot(df_merged_umap, 'UMAP')
    
    fig_properties_pca = create_properties_plot(df_merged_pca, 'PCA')
    fig_properties_umap = create_properties_plot(df_merged_umap, 'UMAP')
    
    fig_target_pca.show()
    fig_target_umap.show()
    fig_properties_pca.show()
    fig_properties_umap.show()
    
    # save the figures
    fig_target_pca.write_html(fig_output_path)
    fig_target_umap.write_html(fig_output_path)
    fig_properties_pca.write_html(fig_output_path)
    fig_properties_umap.write_html(fig_output_path)
    
    return None

def run_analysis(main_df, input_data_path, output_dir='src/data/embeddings_dim_reduction/', 
                 pca_sensitivity=0.95, 
                 umap_random_state=42, 
                 n_embeddings=10, 
                 n_components=2,
                 do_umap=True):
    """
    Run the complete dimensionality reduction pipeline.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    # name is the name of the file, without the extension and without the initial embeddings_ prefix
    name = os.path.basename(input_data_path).replace('embeddings_', '').replace('.csv', '').replace('.zip', '')
    print("Loading data...")
    data  = load_data(input_data_path)
    data['contains NaN'] = data.isnull().sum(axis=1)
    print(data['contains NaN'].value_counts())
    
    # Drop rows with NaN values
    data = data.drop(columns=['contains NaN']).dropna()

    smiles_data = []
    scaler = StandardScaler()
    if "Ligand SMILES" in data.columns:
        smiles_data = data['Ligand SMILES']
        data = data.drop(columns=['Ligand SMILES'])
  
    scaled_descriptors = scaler.fit_transform(data)
    
    print("Running PCA...")
    pca_save_path = os.path.join(output_dir, f'pca_{name}.csv.zip')
    pca_df, pca_result = run_pca_reduction(scaled_descriptors=scaled_descriptors, save_path=pca_save_path, sensitivity=pca_sensitivity, smiles_data=smiles_data)
    
    # Create PCA feature importance plot
    pca_feat_path = os.path.join(output_dir, f'pca_features_{name}.html')
    create_pca_feature_importance_plot(data, pca_result, pca_feat_path, n_embeddings=n_embeddings, n_components=n_components)

    if not do_umap:
        return None
    
    print("Running UMAP...")
    # Run UMAP reduction
    umap_save_path = os.path.join(output_dir, f'umap_{name}.csv.zip')
    umap_df = run_umap_reduction(scaled_descriptors, smiles_data, umap_save_path, random_state=umap_random_state)
    
    # Create visualizations
    fig_output_path = os.path.join(output_dir, f'reduction_plots_{name}.html')
    vizualize_reduction(pca_df, umap_df, main_df, fig_output_path)
    
 

