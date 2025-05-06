import chromadb
from langchain_chroma import Chroma

import src.global_config as global_config

embedding_model = global_config.m3e_large_embed_model
collection_name = "rag_docs"

client_settings = chromadb.config.Settings(
    chroma_server_host="localhost",
    chroma_server_http_port=8000,
    chroma_server_ssl_enabled=False,
    chroma_client_auth_provider="chromadb.auth.token.TokenAuthClientProvider",
    is_persistent=True,
    allow_reset=True,
)
chroma_client = chromadb.Client(client_settings)

chroma_store = Chroma(
    client=chroma_client,
    collection_name=collection_name,
    embedding_function=embedding_model,
)



