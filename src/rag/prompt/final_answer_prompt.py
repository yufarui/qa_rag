from pydantic import BaseModel, Field
from typing_extensions import Optional


class FinalAnswer(BaseModel):
    final_answer: Optional[str] = Field(description="最终的答案")
    page_number: Optional[int] = Field(description="答案所在的页码")
    image_path: Optional[list[str]] = Field(description="答案所在图片的路径")