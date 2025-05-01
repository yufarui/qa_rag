import logging
import pickle
from pathlib import Path

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

# 日志配置
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_bm25_retriever(split_docs: list[Document], k=5):
    # 提取文档中的纯文本内容
    texts = [doc.page_content for doc in split_docs]

    # 创建BM25检索器
    # 保留元数据
    bm25_retriever = BM25Retriever.from_texts(
        texts=texts,
        metadatas=[doc.metadata for doc in split_docs]
    )
    # 设置返回结果数量
    bm25_retriever.k = k
    return bm25_retriever


# 保存BM25数据到磁盘
def save_bm25(split_docs: list[Document], k=5, save_path: str = "bm25_index"):
    retriever = create_bm25_retriever(split_docs, k)
    # 创建目录
    Path(save_path).mkdir(exist_ok=True)

    # 序列化核心数据
    data = {
        "texts": retriever.docs,
        "meta_datas": [doc.metadata for doc in retriever.docs],
        # 核心BM25模型参数
        "bm25_model": retriever.bm25
    }

    # 保存到文件
    with open(f"{save_path}/bm25_data.pkl", "wb") as f:
        pickle.dump(data, f)

    logger.info(f"BM25索引已保存至 {save_path} 目录")
    return retriever


# 从磁盘加载BM25
def load_bm25(load_path: str = "bm25_index") -> BM25Retriever:
    with open(f"{load_path}/bm25_data.pkl", "rb") as f:
        data = pickle.load(f)

    # 重建文档对象
    docs = [
        Document(page_content=text, metadata=metadata)
        for text, metadata in zip(data["texts"], data["meta_datas"])
    ]

    # 重建检索器
    retriever = BM25Retriever.from_documents(docs)

    retriever.bm25 = data["bm25_model"]
    return retriever
