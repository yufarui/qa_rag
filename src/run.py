import os
import logging
from langchain.retrievers import EnsembleRetriever
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableBranch
from typing_extensions import Tuple, List

import constant
import src.global_config as global_config
import src.rag.retriever.faiss_handler as faiss_handler
import src.rag.retriever.bm25_handler as bm25_handler
import src.rag.loader.pdf_parse as pdf_parse
import src.rag.llm.rerank_model as rerank_model
from src.base_model.manual_images import ManualImages
from src.rag.prompt.content_themes_from_prompt import split_theme_prompt_template, ContentResponse
from src.rag.prompt.final_answer_from_prompt import FinalAnswer

# 日志配置
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

llm = global_config.llm
embed_model = global_config.embed_model

split_docs: list[Document] = pdf_parse.load_and_split()

if os.path.exists(constant.faiss_store_path):
    faiss_vector_store = faiss_handler.load_existing_vectorstore()
else:
    faiss_vector_store = faiss_handler.save_vectorstore(split_docs)


# 结合FAISS的混合检索示例
# 创建混合检索器（BM25 + FAISS）
def create_ensemble_retriever():
    # 创建FAISS检索器
    faiss_retriever = faiss_vector_store.as_retriever(
        # 启用相似度阈值过滤
        search_type="similarity_score_threshold",
        search_kwargs={"k": 5, "score_threshold": 0.4}
    )

    # 创建BM25检索器
    bm25_retriever = bm25_handler.create_bm25_retriever(split_docs, k=3)

    # 组合成混合检索器
    ensemble_retriever = EnsembleRetriever(
        retrievers=[faiss_retriever, bm25_retriever],
        # 可以调整权重
        weights=[0.8, 0.2]
    )
    return ensemble_retriever


def format_docs(docs):
    if not docs:
        return None

    return f"{'*' * 10}".join(doc.page_content for doc in docs)


def query_multi_content(question: str) -> ContentResponse:
    split_theme_prompt = PromptTemplate.from_template(split_theme_prompt_template)

    structured_llm = llm.with_structured_output(ContentResponse, method="function_calling")

    # 初始化带过滤的检索器
    ensemble_retriever = create_ensemble_retriever()

    # 定义空响应处理分支
    empty_response = lambda _: ContentResponse(content_list=[])

    # 创建条件分支
    content_check_branch = RunnableBranch(
        # 检查格式化后内容是否为空
        (lambda x: x["context"] is None or not x["context"].strip(), empty_response),
        split_theme_prompt | structured_llm
    )

    rag_chain = {"context": ensemble_retriever | format_docs} | content_check_branch
    result = rag_chain.invoke(question)
    logger.info(f"llm-返回结果 {result}")
    return result


def main(query):
    response: ContentResponse = query_multi_content(query)
    logger.info(response)
    if not response.content_list:
        return

    content_list = [single.extracted_content for single in response.content_list]
    rerank_content: list[str] = rerank_model.predict(query, content_list)
    logger.info(rerank_content)
    docs_with_score: List[Tuple[Document, float]] = faiss_vector_store \
        .similarity_search_with_score(rerank_content[0], k=1)

    page_num, images_path = None, None
    if docs_with_score:
        doc, score = docs_with_score[0]
        page_num = doc.metadata.get("page")
        images_info = doc.metadata.get("images_info")
        if images_info:
            images_path = [ManualImages.model_validate(item).image_path for item in images_info]

    final_answer = FinalAnswer(final_answer=rerank_content[0], image_path=images_path, page_number=page_num)
    logger.info(final_answer)


if __name__ == '__main__':
    main("副仪表台按钮如何操作中央显示屏？")
