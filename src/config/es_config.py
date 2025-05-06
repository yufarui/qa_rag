import os
from dotenv import load_dotenv
from elasticsearch import Elasticsearch
from elasticsearch.exceptions import ConnectionError, AuthenticationException

# 加载 .env 文件中的变量
load_dotenv()

def get_es_client():
    es_host = os.getenv("ES_HOST")
    es_user = os.getenv("ES_USERNAME")
    es_pass = os.getenv("ES_PASSWORD")

    if not es_host:
        raise ValueError("ES_HOST is not defined in .env")

    try:
        # 使用连接池，线程安全（由 Elasticsearch 内部实现）
        if es_user and es_pass:
            es = Elasticsearch(
                hosts=[es_host],
                basic_auth=(es_user, es_pass),
                request_timeout=30,
                max_retries=3,
                retry_on_timeout=True
            )
        else:
            es = Elasticsearch(
                hosts=[es_host],
                request_timeout=30,
                max_retries=3,
                retry_on_timeout=True
            )

        # 测试连接
        if not es.ping():
            raise ConnectionError("Cannot connect to Elasticsearch")

        return es

    except AuthenticationException:
        raise RuntimeError("Elasticsearch authentication failed")
    except ConnectionError as e:
        raise RuntimeError(f"Elasticsearch connection error: {e}")
