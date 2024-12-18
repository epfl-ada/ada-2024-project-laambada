
import matplotlib.pyplot as plt
import seaborn as sns
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

def create_target_plot(df_merged, reduction ) : 
    if reduction == 'PCA':
        fig = px.scatter(df_merged, x='PC1', y='PC2', color='Target Name')
    elif reduction == 'TSNE':
        fig = px.scatter(df_merged, x='TSNE1', y='TSNE2', color='Target Name')

    fig.update_layout(
    height=1200, width=1600, margin=dict(l=20, r=20, t=20, b=20) 
)
    return fig

def create_properties_plot(df_merged, reduction ) : 
    
    fig = go.Figure()

    # Add initial trace with Ki as the color axis
    marker = marker=dict(
                color=df_merged['pKi'],
                colorbar=dict(title='pKi Value'),
                colorscale='Viridis'
            )
    if reduction == 'PCA':
        x_col = 'PC1'
        y_col = 'PC2'
    elif reduction == 'TSNE':
        x_col = 'TSNE1'
        y_col = 'TSNE2'
    
    fig.add_trace(
        go.Scatter(
            x=df_merged[x_col],
            y=df_merged[y_col],
            mode="markers",
            marker=marker
        )
    )

    # Define buttons for switching between Ki and IC50

    buttons = [
        dict(
            label='pKi',
            method='update',
            args=[{
                'marker.color': [df_merged['pKi']],
                'marker.colorbar.title': ['pKi Value']
            }]
        ),
        dict(
            label='pIC50',
            method='update',
            args=[{
                'marker.color': [df_merged['pIC50']],
                'marker.colorbar.title': ['pIC50 Value']
            }]
        )
    ]

    # Add buttons to layout
    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                buttons=buttons,
                direction="down",
                pad={"r": 10, "t": 0},
                showactive=True,
                x=0.17,
                xanchor="left",
                y=1.15,
                yanchor="top"
            )
        ]
    )

    # Update layout to move the legend to the bottom
    fig.update_layout(
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.5,
            xanchor="center",
            x=0.5
        ), 
        xaxis_title=x_col,
        yaxis_title=y_col
    )
    return fig

def reduce_family(value) : 
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
    
def create_pca_feature_importance_plot(data, pca):

    feature_importance = pd.DataFrame(
        pca.components_[:2].T,
        columns=['PC1', 'PC2'],
        index=data.columns
    )

    # Select top contributing features based on absolute values
    top_features = feature_importance.abs().sum(axis=1).sort_values(ascending=False).head(10).index
    top_feature_importance = feature_importance.loc[top_features]

    # Create heatmap 
    fig = go.Figure(data=go.Heatmap(
        z=top_feature_importance.values,
        x=top_feature_importance.columns,
        y=top_feature_importance.index,
        colorscale='RdBu',
        zmid=0
    ))

    fig.update_layout(
        title='PCA Feature Importance Heatmap (Top features)',
        xaxis_title='Principal Component',
        yaxis_title='Feature'
    )

    return fig    