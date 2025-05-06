from elasticsearch import Elasticsearch
from langchain.schema import Document
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStore
from pydantic import Field
from sklearn.preprocessing import MinMaxScaler
import numpy as np


class HybridRetriever(BaseRetriever):
    vector_store: VectorStore
    keyword_store: VectorStore
    top_k: int = Field(default=5, description="Number of documents to return")
    alpha: float = Field(default=0.7, description="vector_store 所占权重比例")

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> list[Document]:
        # 获取两个通道的文档
        vector_results = self.vector_store.similarity_search_with_score(query, k=self.top_k)
        keyword_results = self.keyword_store.similarity_search_with_score(query, k=self.top_k)

        # 解包为文档和分数，空则设默认空列表
        vector_docs, vector_scores = zip(*vector_results) if vector_results else ([], [])
        keyword_docs, keyword_scores = zip(*keyword_results) if keyword_results else ([], [])

        vector_scores = HybridRetriever.normalize_scores(vector_scores)
        keyword_scores = HybridRetriever.normalize_scores(keyword_scores)

        # 为每条文档设置来源标签
        for doc in vector_docs:
            doc.metadata["source"] = "chroma"
        for doc in keyword_docs:
            doc.metadata["source"] = "elastic"

        # 去重并合并分数
        doc_dict = {}
        # 处理向量检索结果
        for doc, score in zip(vector_docs, vector_scores):
            key = doc.page_content
            hybrid_score = self.alpha * score
            if key in doc_dict:
                doc_dict[key].metadata["hybrid_score"] += hybrid_score
            else:
                # 避免修改原始文档
                doc = doc.model_copy()
                doc.metadata["hybrid_score"] = hybrid_score
                doc_dict[key] = doc

        # 处理关键词检索结果
        for doc, score in zip(keyword_docs, keyword_scores):
            key = doc.page_content
            hybrid_score = (1 - self.alpha) * score
            if key in doc_dict:
                doc_dict[key].metadata["hybrid_score"] += hybrid_score
            else:
                # 避免修改原始文档
                doc = doc.model_copy()
                doc.metadata["hybrid_score"] = hybrid_score
                doc_dict[key] = doc

        sorted_docs = sorted(doc_dict.values(), key=lambda d: d.metadata["hybrid_score"], reverse=True)
        return sorted_docs[:self.top_k]

    # 提取分数并归一化
    @classmethod
    def normalize_scores(cls, scores):
        if not scores:
            return []
        # 归一化
        scaler = MinMaxScaler()
        norm_scores = scaler.fit_transform(np.array(scores).reshape(-1, 1)).flatten()
        return norm_scores

    @classmethod
    def create_hybrid_retriever(cls, split_docs: list[Document]) -> "HybridRetriever":
        import src.rag.retriever.es_handler as es_handler
        import src.rag.retriever.chroma_handler as chroma_handler

        vector_store = chroma_handler.chroma_store
        keyword_store = es_handler.es_store

        split_docs = [
            Document(page_content=doc.page_content, metadata=HybridRetriever.clean_metadata(doc.metadata))
            for doc in split_docs
        ]

        vector_store.add_documents(split_docs)
        keyword_store.add_documents(split_docs)

        return HybridRetriever(vector_store=vector_store, keyword_store=keyword_store)

    @classmethod
    def load_hybrid_retriever(cls) -> "HybridRetriever":
        import src.rag.retriever.es_handler as es_handler

        import src.rag.retriever.chroma_handler as chroma_handler
        vector_store = chroma_handler.chroma_store
        keyword_store = es_handler.es_store

        return HybridRetriever(vector_store=vector_store, keyword_store=keyword_store)

    @classmethod
    def clean_metadata(cls, metadata):
        cleaned = {}
        if metadata is None:
            return cleaned
        for key, value in metadata.items():
            if isinstance(value, (str, int, float, bool)):
                cleaned[key] = value
        return cleaned
