import redis
from langchain_community.cache import RedisCache
from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain.globals import set_llm_cache

from src import constant

llm = ChatOpenAI(api_key=constant.API_KEY, base_url=constant.BASE_URL, verbose=True)

embed_model: Embeddings = OpenAIEmbeddings(api_key=constant.API_KEY, base_url=constant.BASE_URL)

redis_client = redis.Redis(host="localhost", port=6379, db=0, decode_responses=True)

set_llm_cache(RedisCache(redis_client))

m3e_large_embed_model = HuggingFaceEmbeddings(model_name=constant.m3e_large_model_path)
