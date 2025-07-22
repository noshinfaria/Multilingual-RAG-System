# from langchain.text_splitter import RecursiveCharacterTextSplitter

# def get_chunks(text, chunk_size=500, chunk_overlap=100):
#     """
#     Splits the input text into overlapping chunks using Bengali and English sentence separators.

#     Parameters:
#     - text (str): The full cleaned text
#     - chunk_size (int): Maximum number of characters per chunk
#     - chunk_overlap (int): Number of overlapping characters between chunks

#     Returns:
#     - List[str]: List of text chunks
#     """
#     splitter = RecursiveCharacterTextSplitter(
#         chunk_size=chunk_size,
#         chunk_overlap=chunk_overlap,
#         separators=["\n", "।", ".", "?", "!"]
#     )
#     return splitter.split_text(text)


import re
from typing import List

def get_chunks(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    sentences = re.split(r'(?<=[।!?])\s+', text)  # split on Bengali sentence enders
    chunks = []
    chunk = ""

    for sentence in sentences:
        if len(chunk) + len(sentence) <= chunk_size:
            chunk += sentence + " "
        else:
            chunks.append(chunk.strip())
            chunk = sentence + " "

    if chunk:
        chunks.append(chunk.strip())

    # Add overlap
    final_chunks = []
    for i in range(0, len(chunks)):
        start = max(i - 1, 0)
        combined = " ".join(chunks[start:i+1])
        final_chunks.append(combined)

    return final_chunks
