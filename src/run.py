import logging

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnableBranch, RunnablePassthrough

import src.global_config as global_config
import src.rag.llm.rerank_model as rerank_model
import src.rag.loader.pdf_parse as pdf_parse
from src.rag.prompt.content_themes_prompt import split_theme_prompt_template, ContentResponse
from src.rag.prompt.hyde_prompt import hyde_prompt
from src.rag.retriever.hybrid_retriever import HybridRetriever

# 日志配置
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

llm = global_config.llm
embed_model = global_config.embed_model


def format_docs(docs):
    if not docs:
        return None

    return f"{'*' * 10}".join(doc.page_content for doc in docs)


def query_multi_content(question: str, retriever: BaseRetriever) -> ContentResponse:
    split_theme_prompt = PromptTemplate.from_template(split_theme_prompt_template)

    structured_llm = llm.with_structured_output(ContentResponse, method="function_calling")

    # 定义空响应处理分支
    empty_response = lambda _: ContentResponse(content_list=[])

    # 创建条件分支
    content_check_branch = RunnableBranch(
        # 检查格式化后内容是否为空
        (lambda x: x["context"] is None or not x["context"].strip(), empty_response),
        split_theme_prompt | structured_llm
    )

    rag_chain = {"context": retriever | format_docs} | content_check_branch
    result = rag_chain.invoke(question)
    logger.info(f"llm-返回结果 {result}")
    return result


def query_hyde(question: str) -> str:
    hyde_prompt_template = PromptTemplate.from_template(hyde_prompt)

    rag_chain = ({"question": RunnablePassthrough()}
                 | hyde_prompt_template
                 | llm
                 | StrOutputParser())

    result = rag_chain.invoke(question)
    return result


def answer_question(question: str):
    init = True

    if init:
        split_docs = pdf_parse.load_and_split()
        hybrid_retriever = HybridRetriever.create_hybrid_retriever(split_docs)
    else:
        hybrid_retriever = HybridRetriever.load_hybrid_retriever()

    response: ContentResponse = query_multi_content(question, hybrid_retriever)
    logger.info(f"基于文档提取的信息{response}")

    if not response.content_list:
        return
    hyde_answer = query_hyde(question)
    content_list = [info.extracted_content for info in response.content_list]
    content_list.append(hyde_answer)

    rerank_content: list[str] = rerank_model.predict(question, content_list)
    logger.info(f"最后排序的顺序{rerank_content}")


if __name__ == '__main__':
    answer_question("如何通过中央显示屏进行副驾驶员座椅设置？")