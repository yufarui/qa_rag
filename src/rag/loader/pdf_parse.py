import re

import fitz
import tiktoken
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pymongo.collection import Collection
from typing_extensions import List

import src.rag.llm.m3e_small_model as m3e_small_model
from src import constant
from src.base_model.manual_images import ManualImages
from src.base_model.manual_info_mongo import ManualInfo
from src.config.mongodb_config import MongoConfig
import src.rag.loader.image_handler as image_handler

# 公共配置区
encoding = tiktoken.get_encoding("cl100k_base")
manual_text_collection: Collection = MongoConfig.get_collection("manual_text")
file_path = constant.pdf_path

# ===== TextSplitter 设置 =====

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=126,
    # 按这个优先级递归切
    separators=["\n\n"],
    length_function=lambda text: len(encoding.encode(text))
)


# ===== 文本预处理部分 =====

def sentence_split(text: str) -> list[str]:
    """按中文/英文标点切句"""
    sentences = re.split(r'(?<=[。\n\t])+', text.strip())
    return [s.strip() for s in sentences if s.strip()]


def load_pdf() -> list[Document]:
    pdf = fitz.open(file_path)
    raw_docs = []

    for page_num in range(len(pdf)):
        page = pdf.load_page(page_num)
        text = page.get_text()
        images = page.get_images(full=True)

        manual_images_list: List[ManualImages] = []
        for img_index, img in enumerate(images):
            manual_image: ManualImages = image_handler.handle_image(img, img_index, page)
            if manual_image:
                manual_images_list.append(manual_image)

        if text.strip():
            unique_id = f"{hash(text)}_{page}"
            metadata = {
                "unique_id": unique_id,
                "source": file_path,
                "page": page_num + 1,
                "images_info": manual_images_list
            }

            raw_docs.append(Document(page_content=text, metadata=metadata))

    return raw_docs


def load_and_split() -> list[Document]:
    """加载 PDF 文档，进行句子级 + 语义感知切分"""
    raw_docs: list[Document] = load_pdf()
    all_split_docs = []

    for doc in raw_docs:
        sentences = [doc.page_content]

        grouped_chunks = m3e_small_model.semantic_group(sentences, group_size=5)

        for chunk in grouped_chunks:
            # 以 chunk 为单位继续用 langchain 切分（带 overlap）
            split_docs = text_splitter.create_documents([chunk], metadatas=[doc.metadata])

            # save_2_mongo(split_docs)
            all_split_docs.extend(split_docs)

    return all_split_docs


def save_2_mongo(split_docs):
    for doc in split_docs:
        # 从 metadata 中提取关键参数
        metadata = doc.metadata
        page = metadata.get("page")

        # 构造唯一性 unique_id
        unique_id = metadata.get("unique_id")
        # 处理 images_info 字段
        images_info = metadata.get("images_info")

        # 创建文档记录对象
        doc_record = ManualInfo(
            unique_id=unique_id,
            page=page,
            related_content=doc.page_content,
            images_info=images_info
        )

        # 更新数据库操作
        manual_text_collection.update_one(
            {"unique_id": doc_record.unique_id},
            {"$set": doc_record.model_dump()},
            upsert=True
        )
