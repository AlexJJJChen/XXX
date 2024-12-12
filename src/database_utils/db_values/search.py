import pickle
from datasketch import MinHash, MinHashLSH
from pathlib import Path
import logging
from typing import Dict, Tuple, List

from database_utils.db_values.preprocess import _create_minhash

### Database value similarity ###

def _jaccard_similarity(m1: MinHash, m2: MinHash) -> float:
    """
    Computes the Jaccard similarity between two MinHash objects.

    Args:
        m1 (MinHash): The first MinHash object.
        m2 (MinHash): The second MinHash object.

    Returns:
        float: The Jaccard similarity between the two MinHash objects.
    """
    return m1.jaccard(m2)

def load_db_lsh(db_directory_path: str) -> Tuple[MinHashLSH, Dict[str, Tuple[MinHash, str, str, str]]]:
    """
    Loads the LSH and MinHashes from the preprocessed files in the specified directory.

    Args:
        db_directory_path (str): The path to the database directory.

    Returns:
        Tuple[MinHashLSH, Dict[str, Tuple[MinHash, str, str, str]]]: The LSH object and the dictionary of MinHashes.

    Raises:
        Exception: If there is an error loading the LSH or MinHashes.
    """
    db_id = Path(db_directory_path).name
    try:
        with open(Path(db_directory_path) / "preprocessed" / f"{db_id}_lsh.pkl", "rb") as file:
            lsh = pickle.load(file)
        with open(Path(db_directory_path) / "preprocessed" / f"{db_id}_minhashes.pkl", "rb") as file:
            minhashes = pickle.load(file)
        return lsh, minhashes
    except Exception as e:
        logging.error(f"Error loading LSH for {db_id}: {e}")
        raise e

def query_lsh(lsh: MinHashLSH, minhashes: Dict[str, Tuple[MinHash, str, str, str]], keyword: str, 
              signature_size: int = 20, n_gram: int = 3, top_n: int = 10) -> Dict[str, Dict[str, List[str]]]:
    """
    Queries the LSH for similar values to the given keyword and returns the top results.

    Args:
        lsh (MinHashLSH): The LSH object.
        minhashes (Dict[str, Tuple[MinHash, str, str, str]]): The dictionary of MinHashes.
        keyword (str): The keyword to search for.
        signature_size (int, optional): The size of the MinHash signature.
        n_gram (int, optional): The n-gram size for the MinHash.
        top_n (int, optional): The number of top results to return.

    Returns:
        Dict[str, Dict[str, List[str]]]: A dictionary containing the top similar values.
    """
    query_minhash = _create_minhash(signature_size, keyword, n_gram)
    results = lsh.query(query_minhash)

    similarities = [(result, _jaccard_similarity(query_minhash, minhashes[result][0])) for result in results]
    similarities = sorted(similarities, key=lambda x: x[1], reverse=True)[:top_n]

    similar_values_trimmed: Dict[str, Dict[str, List[str]]] = {}
    for result, similarity in similarities:
        table_name, column_name, value = minhashes[result][1:]
        if table_name not in similar_values_trimmed:
            similar_values_trimmed[table_name] = {}
        if column_name not in similar_values_trimmed[table_name]:
            similar_values_trimmed[table_name][column_name] = []
        similar_values_trimmed[table_name][column_name].append(value)

    return similar_values_trimmed


# import pickle
# import numpy as np
# from pathlib import Path
# import logging
# from typing import Dict, Tuple, List, Any
# from sklearn.feature_extraction.text import TfidfVectorizer

# ### Database value similarity ###

# def _cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
#     """
#     Computes the Cosine similarity between two vectors.

#     Args:
#         vec1 (np.ndarray): The first vector.
#         vec2 (np.ndarray): The second vector.

#     Returns:
#         float: The Cosine similarity between the two vectors.
#     """
#     return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

# def load_db_lsh(db_directory_path: str) -> Tuple[Any, Dict[str, Tuple[np.ndarray, str, str, str]]]:
#     """
#     Loads the vector representations from the preprocessed files in the specified directory.

#     Args:
#         db_directory_path (str): The path to the database directory.

#     Returns:
#         Tuple[Any, Dict[str, Tuple[np.ndarray, str, str, str]]]: A dummy LSH object (for compatibility) and the dictionary of vectors.

#     Raises:
#         Exception: If there is an error loading the vectors.
#     """
#     db_id = Path(db_directory_path).name
#     try:
#         with open(Path(db_directory_path) / "preprocessed" / f"{db_id}_vectors.pkl", "rb") as file:
#             vectors = pickle.load(file)
#         return None, vectors
#     except Exception as e:
#         logging.error(f"Error loading vectors for {db_id}: {e}")
#         raise e

# def query_lsh(lsh: Any, vectors: Dict[str, Tuple[np.ndarray, str, str, str]], keyword: str, 
#               signature_size: int = 20, n_gram: int = 3, top_n: int = 10) -> Dict[str, Dict[str, List[str]]]:
#     """
#     Queries the vector representations for similar values to the given keyword and returns the top results.

#     Args:
#         lsh (Any): The dummy LSH object.
#         vectors (Dict[str, Tuple[np.ndarray, str, str, str]]): The dictionary of vectors.
#         keyword (str): The keyword to search for.
#         signature_size (int, optional): This parameter is no longer used.
#         n_gram (int, optional): This parameter is no longer used.
#         top_n (int, optional): The number of top results to return.

#     Returns:
#         Dict[str, Dict[str, List[str]]]: A dictionary containing the top similar values.
#     """
#     # Generate the vector for the query keyword
#     vectorizer = TfidfVectorizer()
#     query_vector = vectorizer.transform([keyword]).toarray()[0]
    
#     similarities = [(result, _cosine_similarity(query_vector, vectors[result][0])) for result in vectors]
#     similarities = sorted(similarities, key=lambda x: x[1], reverse=True)[:top_n]

#     similar_values_trimmed: Dict[str, Dict[str, List[str]]] = {}
#     for result, similarity in similarities:
#         table_name, column_name, value = vectors[result][1:]
#         if table_name not in similar_values_trimmed:
#             similar_values_trimmed[table_name] = {}
#         if column_name not in similar_values_trimmed[table_name]:
#             similar_values_trimmed[table_name][column_name] = []
#         similar_values_trimmed[table_name][column_name].append(value)

#     return similar_values_trimmed