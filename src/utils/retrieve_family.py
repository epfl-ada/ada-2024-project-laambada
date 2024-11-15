'''
File name: retrieve_family.py
Author: Alexandre Sallinen
Date created: 14 November 2024
Date last modified: 15 November 2024
Python Version: 3.7
'''
import aiohttp
import asyncio
from tqdm import tqdm
from typing import List, Dict
import logging

async def fetch_protein_family(session: aiohttp.ClientSession, protein_id: str) -> tuple:
    '''
    Fetch protein family information from the Uniprot API.
    '''
    url = f"https://rest.uniprot.org/uniprotkb/search?query={protein_id}&fields=xref_panther"
    try:
        async with session.get(url) as response:
            if response.status == 200:
                data = await response.json()
                families = []
                for result in data.get("results", []):
                    for ref in result.get("uniProtKBCrossReferences", []):
                        if ref.get("database") == "PANTHER" and ":" not in ref.get("id", ""):
                            for prop in ref.get("properties", []):
                                if prop.get("key") == "EntryName":
                                    families.append(prop.get("value"))
                return protein_id, families
            return protein_id, []
    except Exception as e:
        logging.error(f"Error fetching {protein_id}: {response.status}")
        return protein_id, []

async def retrieve_family(id_list: List[str]) -> Dict[str, list]:
    '''
    Asynchronously retrieve family information for multiple proteins.
    '''
    async with aiohttp.ClientSession(
        connector=aiohttp.TCPConnector(limit=50),  # Connection pooling
        timeout=aiohttp.ClientTimeout(total=30)
    ) as session:
        tasks = [fetch_protein_family(session, pid) for pid in id_list]
        results = []
        for future in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
            results.append(await future)
        
        return {pid: families for pid, families in results if families}

def get_protein_families(id_list: List[str]) -> Dict[str, list]:
    '''
    Synchronous wrapper for the async function.
    '''
    return asyncio.run(retrieve_family(id_list))