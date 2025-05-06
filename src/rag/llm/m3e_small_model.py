import math
from typing import List
import pandas as pd

import torch
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering

from src import constant

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize model with GPU support and normalization
embedding_model = SentenceTransformer("moka-ai/m3e-small")


def semantic_group(sentences: List[str], group_size: int = 5) -> List[str]:
    """将句子按语义相似性分组

    Args:
        sentences: 待分组的句子列表
        group_size: 每组的目标最大句子数（实际可能略多）

    Returns:
        合并后的分组文本列表

    Raises:
        ValueError: 当输入参数不合法时
    """
    # 参数校验
    if group_size < 1:
        raise ValueError("group_size must be at least 1")
    if not sentences:
        return []

    # 当数据量不足时直接返回
    if len(sentences) <= group_size:
        return [" ".join(sentences)]

    # 计算合理的聚类数量（向上取整）
    n_clusters = max(1, math.ceil(len(sentences) / group_size))

    # 生成嵌入向量（已自动使用GPU加速）
    embeddings = embedding_model.encode(sentences)

    # 使用余弦相似度的层次聚类
    clustering = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric="cosine",  # 使用余弦距离
        linkage="average",  # 使用平均链接算法
        compute_full_tree="auto"
    )

    try:
        labels = clustering.fit_predict(embeddings)
    except Exception as e:
        raise RuntimeError(f"Clustering failed: {str(e)}") from e

    df = pd.DataFrame({"sentence": sentences, "label": labels})

    result = (df.groupby("label", sort=True)['sentence']
              .agg(lambda x: " ".join(x))
              .to_dict())
    return list(result.values())
