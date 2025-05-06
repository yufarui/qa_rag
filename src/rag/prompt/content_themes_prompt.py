from typing_extensions import Optional, Literal

from pydantic import BaseModel, Field

from src.base_model.manual_images import ManualImages

split_theme_prompt_template = """
==== 任务说明 ====
以下是几段从知识库中检索出的文档，它们可能来自不同的来源，内容可能存在重复、冗余或逻辑混乱的问题。你的任务是：\n"
"1. 对这些文档的内容进行整合和归纳，总结出清晰的主题。\n"
"2. 根据主题提炼出文档的关键内容，并去掉重复的信息，对于内容相似的文档可以进行取舍与合并。\n"
"3. 将文档重新组织为多条逻辑清晰的文章，条理分明，语言简洁。\n"

==== 输出要求 ====
* 必须使用JSON格式返回，符合预定义模式
* 请务必按照整理的文档,进行汇总
* 重复的文档可以合并
* 如无有效内容，返回空

==== 以下是需要整理的文档 ===
{context}
"""


class SingleContent(BaseModel):
    """
    单条内容结构
    """
    extracted_content: str = Field(description="提取的内容,控制在100-200字")
    keyword: Optional[str] = Field(description="关键字")


class ContentResponse(BaseModel):
    """
    总体返回的数据结构
    """

    content_list: Optional[list[SingleContent]] \
        = Field(description="提取的内容列表")
