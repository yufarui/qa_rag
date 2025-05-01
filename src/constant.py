from src.load_env import get_env_key

# 路径都以 run.py 为参考路径
image_save_dir = "../manual_images"
pdf_path = "../data/train_a.pdf"
faiss_store_path = "manual_text_faiss_index"

API_KEY = get_env_key("ZHI_ZENG_API_KEY")
BASE_URL = get_env_key("BASE_URL")

#  模型路径
rerank_model_path = "../pre_train_model/bge-reranker-large"
m3e_large_model_path = "../pre_train_model/bge-reranker-large"