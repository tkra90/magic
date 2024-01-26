from typing import List, Union
import chromadb


def wrap_text(txt: str, n_char: int = 50, separator: str = "\n"):
    """
    Wrap a text at the next space after n_char characters.

    Parameters:
    - txt (str): The input text to be wrapped.
    - n_char (int, optional): The maximum number of characters before wrapping.
    - separator (str): The separator added after each wrapped line

    Returns:
    - str: The wrapped text.
    """
    if len(txt) < n_char:
        return txt
    else:
        last_space_index = txt[:n_char].rfind(" ")
        wrapped_line = txt[:last_space_index].strip() + separator
        remaining_text = txt[last_space_index + 1 :]
    return wrapped_line + wrap_text(remaining_text, n_char)


def query_chroma_collection(
    query: str, chroma_col: chromadb.api.Collection, n: int = 5
) -> dict:
    """
    Query the ChromaDB, a helper function.
    Parameters:
    - query (str): The input text to query the DB.
    - chroma_col (chromadb.api.Collection): The ChromaDB collection.
    - n (int): number of results to generate (default = 5)

    Returns:
    - dict: A dictionary with the results (documents and embeddings).
    """
    res = chroma_col.query(
        query_texts=query, n_results=n, include=["documents", "embeddings"]
    )
    return res


def wrap_chroma_results(doc: Union[List[str], str], n_char: int = 50) -> str:
    """
    Wrap results of a ChromaDB colection query.
    Parameters:
    - doc (Union[List [str], str]): The input can be list of string or a string.
    - n_char (int, optional): The maximum number of characters before wrapping.

    Returns:
    - str: A wrapped result of the query
    """
    if isinstance(doc, list):
        res = ""
        for d in doc:
            res += wrap_text(d, n_char) + "\n\n"
        return res
    else:
        return wrap_text(doc, n_char)
