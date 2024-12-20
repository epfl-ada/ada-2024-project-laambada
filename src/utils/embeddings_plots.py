import matplotlib.pyplot as plt
import seaborn as sns
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

def create_target_plot(df_merged, reduction ) : 
    '''
    Create a scatter plot for the dimensionality (UMAP or PCA) reduction visualization.
    The plot is colored by the Protein target name for each point.

    Parameters:
    - df_merged (pd.DataFrame): DataFrame containing the dimensionality reduction results
    - reduction (str): Type of reduction to use for the plot

    Returns:
    - go.Figure: Plotly figure object
    '''
    if reduction == 'PCA':
        fig = px.scatter(df_merged, x='PC1', y='PC2', color='Target Name')

    elif reduction == 'UMAP':
        fig = px.scatter(df_merged, x='UMAP1', y='UMAP2', color='Target Name')

    fig.update_layout(
    height=1200, width=1600, margin=dict(l=20, r=20, t=20, b=20) 
)
    return fig

def create_properties_plot(df_merged, reduction ):
    '''
    Crate a scatter plot for the dimensionality (UMAP or PCA) reduction visualization.
    The plot is colored by the Protein target name for each point.

    Parameters:
    - df_merged (pd.DataFrame): DataFrame containing the chemical properties
    - reduction (str): Type of reduction to use for the plot

    Returns:
    - go.Figure: Plotly figure object
    '''
    df_pKi = df_merged.dropna(subset=['pKi'])
    df_pIC = df_merged.dropna(subset=['pIC50'])

    # 1. For pKi
    fig_pKi = go.Figure()

    # Add initial trace with Ki as the color axis
    marker = marker=dict(
                color=df_pKi['pKi'],
                colorbar=dict(title='pKI Value'),
                colorscale='PuRd'
            )
    if reduction == 'PCA':
        x_col = 'PC1'
        y_col = 'PC2'
    elif reduction == 'UMAP':
        x_col = 'UMAP1'
        y_col = 'UMAP2'
    
    fig_pKi.add_trace(
        go.Scatter(
            x=df_pKi[x_col],
            y=df_pKi[y_col],
            mode="markers",
            marker=marker
        )
    )

    fig_pKi.update_layout(
        xaxis_title=x_col,
        yaxis_title=y_col
    )

    # 2. For pIC50
    fig_pIC = go.Figure()
    marker = marker=dict(
                color=df_pIC['pIC50'],
                colorbar=dict(title='pIC50 Value'),
                colorscale='PuRd'
            )
    if reduction == 'PCA':
        x_col = 'PC1'
        y_col = 'PC2'
    elif reduction == 'UMAP':
        x_col = 'UMAP1'
        y_col = 'UMAP2'
    
    fig_pIC.add_trace(
        go.Scatter(
            x=df_pIC[x_col],
            y=df_pIC[y_col],
            mode="markers",
            marker=marker
        )
    )

    fig_pIC.update_layout(
        xaxis_title=x_col,
        yaxis_title=y_col
    )

    return fig_pKi, fig_pIC


def reduce_family(value) : 
    '''
    Simplify the protein target name, by removing the specific mutants to group the similar proteins together.

    Parameters:
    - value (str): Protein target name

    Returns:
    - str: Reduced protein target name
    '''
    conditions_met = 0
    result = value
    if 'Tyrosine-protein kinase JAK2' in value : 
        result =  'Tyrosine-protein kinase JAK2'
        conditions_met += 1
    if 'Tyrosine-protein kinase JAK3' in value : 
        result =  'Tyrosine-protein kinase JAK3'
        conditions_met += 1
    if 'Tyrosine-protein kinase JAK1' in value : 
        result = 'Tyrosine-protein kinase JAK1'
        conditions_met += 1
    if 'Non-receptor tyrosine-protein kinase TYK2' in value : 
        result = 'Non-receptor tyrosine-protein kinase TYK2'
        conditions_met += 1
    if 'Tyrosine-protein kinase Mer' in value : 
        result =  'Tyrosine-protein kinase Mer'
        conditions_met += 1
    if "cAMP-specific 3',5'-cyclic phosphodiesterase 4A" in value : 
        result =  "cAMP-specific 3',5'-cyclic phosphodiesterase 4A"
        conditions_met += 1
    if "Glutathione S-transferase P" in value :
        conditions_met += 1
    if conditions_met > 1:
        return value
    else :
        return result
    