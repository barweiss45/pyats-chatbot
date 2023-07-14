import chromadb
from chromadb.config import Settings
client = chromadb.Client(Settings(
    chroma_db_impl="duckdb+parquet",
    persist_directory="./.chromadb/" # Optional, defaults to .chromadb/ in the current directory
))

collection = client.get_collection(name="pyats_docs")

print(collection.peek()) # returns a list of the first 10 items in the collection
print(collection.count()) # returns the number of items in the collection
# collection.modify(name="new_name") # Rename the collection
# resets entire database - this *cant* be undone!
# client.reset()

# returns timestamp to check if service is up
# client.heartbeat()
"""
        collection.get()
        >>> type(collection.get())
        <class 'dict'>
        >>> for key in collection.get().keys():
        ...     print(key)
        ... 
        ids
        embeddings
        documents
        metadatas
        >>>
""" 