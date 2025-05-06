from elasticsearch import Elasticsearch
from elasticsearch.helpers.vectorstore import BM25Strategy
from langchain_elasticsearch import ElasticsearchStore

# 初始化 Elasticsearch 客户端
es_client = Elasticsearch("http://localhost:9200")
index_name = "rag_docs"

# 构建 ElasticsearchStore
es_store = ElasticsearchStore(
    es_url="http://localhost:9200",
    index_name=index_name,
    strategy=BM25Strategy(),
)
