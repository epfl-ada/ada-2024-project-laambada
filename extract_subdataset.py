import pandas as pd
import zipfile

zip_file_path = 'data/BindingDB_All_202411_tsv.zip' 
tsv_file_name = 'BindingDB_All.tsv' 
 
# Open the zip file 
with zipfile.ZipFile(zip_file_path, 'r') as z: 
    # Open the TSV file within the zip 
    with z.open(tsv_file_name) as file: 
        # Read the TSV file into a DataFrame 
        df = pd.read_csv(file, sep='\t', on_bad_lines='skip') 
 
# Randomly sample 1000 rows
sample_df = df.sample(n=1000, random_state=42)

# Save the sample to a new tsv file
sample_df.to_csv('data/BindingDB_sample.tsv', sep='\t', index=False)