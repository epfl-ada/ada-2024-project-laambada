'''
File name: 
Author: Aygul Bayramova
Date created: 14 November 2024
Date last modified: 14 November 2024
Python Version: 3.7
'''

def describe_column(df, column_name, print_types=True, print_uniques_values=True, print_missing_values=True):
    '''
    This function prints the types of data, number of unique values, 
    and number of missing values in a column of a dataframe.
    
    :param df: The dataframe to describe
    :param column_name: The name of the column to describe
    :param print_types: Whether to print the types of data in the column
    :param print_uniques_values: Whether to print the number of unique values in the column
    :param print_missing_values: Whether to print the number of missing values in the column

    :return: None
    '''
    if column_name not in df.columns:
        raise ValueError("The column name is not in the dataframe")
    if print_types:
        print("Types of data in " + column_name + f" column: {df[column_name].dtypes}")
    if print_uniques_values:
        print("Number of unique values in " + column_name + f" column: {df[column_name].nunique()}")
    if print_missing_values:
        print("Number of missing values in " + column_name + f" column: {df[column_name].isnull().sum()}")
    return