import logging
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
import src.global_config as global_config

# 日志配置
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def save_vectorstore(split_docs: list[Document]):
    # 假设 split_docs 是你的文档列表
    faiss_vector_store = FAISS.from_documents(
        documents=split_docs,
        embedding=global_config.embed_model,
    )

    # 保存到本地磁盘
    faiss_vector_store.save_local("manual_text_faiss_index")
    logger.info("FAISS 索引已保存至 faiss_index 文件夹")
    return faiss_vector_store


# 从磁盘加载 FAISS 向量存储
def load_existing_vectorstore():
    # 加载时需要提供相同的嵌入模型
    loaded_vector_store = FAISS.load_local(
        folder_path="manual_text_faiss_index",
        embeddings=global_config.embed_model,
        # 必要的安全确认参数
        allow_dangerous_deserialization=True,
    )
    logger.info("FAISS 索引已从磁盘加载")
    return loaded_vector_store


def faiss_retriever_with_score(faiss_vector_store: FAISS, query: str, k: int = 10, threshold: float = 0.6):
    # 获取文档及相似度分数
    docs_with_scores = faiss_vector_store.similarity_search_with_score(query, k=k)
    docs_with_scores.sort(key=lambda x: x[1])
    # 根据分数过滤
    filtered_docs = [doc for doc, cos_sim in docs_with_scores if cos_sim <= 1 - threshold]
    return filtered_docs
