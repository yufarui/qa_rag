from pydantic import BaseModel, Field
from typing_extensions import Optional


class FinalAnswer(BaseModel):
    page_number: Optional[int] = Field(description="答案所在的页码")
    image_path: Optional[str] = Field(description="答案所在图片的路径")

    def find_page_number(self, metadata: dict) -> int:
        """
        查询答案所在的页码
        """
        pass

    def find_image_path(self,  metadata: dict) -> str:
        """
        查询答案所在图片的路径
        """
        pass