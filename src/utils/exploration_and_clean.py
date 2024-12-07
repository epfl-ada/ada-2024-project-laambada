import pandas as pd

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
