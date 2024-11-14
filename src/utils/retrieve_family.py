'''
File name: retrieve_family.py
Author: Aygul Bayramova
Date created: 14 November 2024
Date last modified: 14 November 2024
Python Version: 3.7
'''
import requests
from tqdm import tqdm

def retrieve_family(id_list):
    '''
    This function retrieves the family information of the proteins
    from the Uniprot database using the Uniprot API.
    :param id_list: A list of UniProt IDs
    :return: A dictionary where the keys are the UniProt IDs and the values are the family names
    '''
    false_count = 0
    family_dict = {}
    for test_id in tqdm(id_list):
        response = requests.get(
            f"https://rest.uniprot.org/uniprotkb/search?query={test_id}&fields=xref_panther",
            headers={"Accept": "application/json"}
        )
        if response.status_code == 200:
            results = response.json().get("results", [])
            for dic in results:
                curr = dic.get("uniProtKBCrossReferences", False)
                if curr:
                    for ref in curr:
                        if ref.get("database") == "PANTHER":
                            if not (":" in ref.get("id") ): # Check if this is a sub family or family
                                properties = ref.get("properties")
                                for el in properties:
                                    if el.get("key") == "EntryName":
                                        family_dict[test_id] = el.get('value')
        else:
            print(f"Error: {response.status_code}, {response.text}")
            false_count += 1

    print(f"Number of failed requests: {false_count}")
    return family_dict