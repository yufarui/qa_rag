# mongo_config.py
import os
from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ConfigurationError

# 加载环境变量
load_dotenv()


class MongoConfig:
    # 默认配置
    _host = os.getenv("MONGO_HOST", "localhost")
    _port = int(os.getenv("MONGO_PORT", 27017))
    _db_name = os.getenv("MONGO_DB_NAME", "mydatabase")
    _username = os.getenv("MONGO_USERNAME")
    _password = os.getenv("MONGO_PASSWORD")
    _auth_source = os.getenv("MONGO_AUTH_SOURCE", "admin")

    # 连接参数
    _max_pool_size = 100
    _connect_timeout = 5000  # 毫秒
    _socket_timeout = 3000  # 毫秒

    # 单例模式
    _client = None
    _db = None

    @classmethod
    def _build_connection_uri(cls):
        """构建MongoDB连接URI"""
        if cls._username and cls._password:
            return f"mongodb://{cls._username}:{cls._password}@{cls._host}:{cls._port}/?authSource={cls._auth_source}"
        return f"mongodb://{cls._host}:{cls._port}"

    @classmethod
    def initialize(cls):
        """初始化MongoDB连接"""
        if cls._client is None:
            try:
                cls._client = MongoClient(
                    cls._build_connection_uri(),
                    maxPoolSize=cls._max_pool_size,
                    connectTimeoutMS=cls._connect_timeout,
                    socketTimeoutMS=cls._socket_timeout,
                    serverSelectionTimeoutMS=5000
                )

                # 验证连接
                cls._client.admin.command('ping')
                cls._db = cls._client[cls._db_name]
                print("Successfully connected to MongoDB")

            except ConfigurationError as e:
                raise RuntimeError(f"MongoDB configuration error: {str(e)}")
            except ConnectionFailure as e:
                raise RuntimeError(f"Failed to connect to MongoDB: {str(e)}")
            except Exception as e:
                raise RuntimeError(f"Unexpected MongoDB connection error: {str(e)}")

    @classmethod
    def get_db(cls):
        """获取数据库实例"""
        if cls._client is None:
            cls.initialize()
        return cls._db

    @classmethod
    def get_collection(cls, collection_name):
        """获取集合实例"""
        return cls.get_db()[collection_name]

    @classmethod
    def close(cls):
        """关闭所有连接"""
        if cls._client:
            cls._client.close()
            cls._client = None
            cls._db = None
            print("MongoDB connection closed")


# 应用启动时初始化连接
MongoConfig.initialize()