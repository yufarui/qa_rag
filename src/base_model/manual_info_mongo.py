from typing import Optional

from pydantic import BaseModel, Field

from src.base_model.manual_images import ManualImages


class ManualInfo(BaseModel):
    unique_id: str = Field(description="唯一标识符")
    page: int = Field(ge=1, description="页码从1开始")
    images_info: Optional[list[ManualImages]] = Field(description="存储的图片信息")
    related_content: Optional[str] = Field(
        description="相关内容，多个区块用换行符连接"
    )
