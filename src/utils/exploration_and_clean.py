import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def explore_column(df, i, return_unique = False):
    columns = df.columns
    column_name = columns[i]
    print(f'# {column_name}')

    nb_nan = df[column_name].isna().sum()
    print(f'{nb_nan} nan values ({round(nb_nan/len(df)*100,2)}%)')

    unique_values = df[column_name].unique()
    nb_unique_values = len(unique_values)
    print(f'{nb_unique_values} unique values')
    if nb_unique_values < 25:
        print(unique_values)

    if return_unique:
        return unique_values
    
def describe_column(df, column_name, print_types=True, print_uniques_values=True, print_missing_values=True):
    if column_name not in df.columns:
        raise ValueError("The column name is not in the dataframe")
    if print_types:
        print("Types of data in " + column_name + f" column: {df[column_name].apply(lambda x: type(x)).unique()}")
    if print_uniques_values:
        print("Number of unique values in " + column_name + f" column: {df[column_name].nunique()}")
    if print_missing_values:
        print("Number of missing values in " + column_name + f" column: {df[column_name].isnull().sum()}")
    return


def quick_check_column(df, i, treshold = 100):
    columns = df.columns
    column_name = columns[i]
    nb_nan = df[column_name].isna().sum()
    percent_nan = nb_nan/len(df)*100
    if percent_nan < treshold:
        print(f'# {column_name}')
        print(f'{nb_nan} nan values ({round(nb_nan/len(df)*100,2)}%)')
        unique_values = df[column_name].unique()
        nb_unique_values = len(unique_values)
        print(f'{nb_unique_values} unique values')
        if nb_unique_values < 25:
            print(unique_values)


def clean_metrics(column) : 
    # Some strings have blank space and < > 
    # Remove any character that is not number + the blank space but keep if we have >< signes to not replace >100 by 100
    clean_column = column.str.replace(r'[^0-9.\-<>]', '', regex=True).str.strip()

    # the <> will be set as nan - see latter if we want to use them
    clean_numerical_column = pd.to_numeric(clean_column, errors= 'coerce')

    return clean_numerical_column 

def plot_chemical_property_distributions(df, df_embeddings, metrics, chemical_properties, properties_colors, title):
    """
    Plot the distribution of chemical properties and additional metrics for ligands.

    Parameters:
    - df (pd.DataFrame): Main DataFrame containing ligand data.
    - df_embeddings (pd.DataFrame): DataFrame containing RDKit-derived chemical properties.
    - metrics (list): List of metric column names to plot (e.g., pKi, pIC50).
    - chemical_properties (list): List of chemical properties or features to visualize.
    - properties_colors (list): List of colors for the histogram plots of chemical properties.
    - title (str): Title of the overall plot.

    Returns:
    - None. Displays the generated Plotly figure.
    """
    # Calculate number of rows and columns for subplots
    n_metrics = len(metrics)
    n_properties = len(chemical_properties)
    
    # Create the subplot grid
    fig = make_subplots(rows=n_metrics, cols=n_properties+1)

    # Plot chemical properties histograms
    for i, metric in enumerate(metrics):
        # Merge main dataframe with embeddings
        df_to_plot = df.merge(df_embeddings, on='Ligand SMILES', how='left')

        for j, property in enumerate(chemical_properties):
            color = properties_colors[j % len(properties_colors)]  # Cycle through colors
            x = df_to_plot[property]
            fig.add_trace(
                go.Histogram(x=x, name=property, showlegend=False, marker_color=color),
                row=i+1, col=j+1
            )
            fig.update_xaxes(title_text=property, row=i+1, col=j+1)

        # Plot metric histogram in the last column
        x = df_to_plot[metric]
        fig.add_trace(
            go.Histogram(x=x, name=metric, showlegend=False, marker_color='wheat'),
            row=i+1, col=n_properties+1
        )
        fig.update_xaxes(title_text=metric, row=i+1, col=n_properties+1)

    # Update layout
    fig.update_layout(
        height=300 * n_metrics,  # Adjust height dynamically
        width=1200,
        title_text=title
    )

    # Show the figure
    fig.show()