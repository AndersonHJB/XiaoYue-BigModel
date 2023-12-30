from pymilvus import connections, utility, Collection, db, CollectionSchema, FieldSchema, DataType
import config


class Milvus:
    def __init__(self):
        super(Milvus, self).__init__()

        self.configs = config.ConfigParser()

        # specify milvus connection
        self.connections = connections.connect(
            host=self.configs.get(key='milvus')['host'],
            port=self.configs.get(key='milvus')['port'],
            user=self.configs.get(key='milvus')['user'],
            password=self.configs.get(key='milvus')['password'],
        )

    @staticmethod
    def create_collection(collection_name, dim):
        # 1. 声明 Field 类型及属性
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
            FieldSchema(name="question", dtype=DataType.VARCHAR, max_length=500),
            FieldSchema(name="answer", dtype=DataType.VARCHAR, max_length=3000),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim)
        ]

        # 2. 根据字段声明 Schema
        schema = CollectionSchema(
            fields=fields,
            description="",
            enable_dynamic_field=True
        )

        # 3. 根据 Schema 声明 Collection
        collection = Collection(
            name=collection_name,
            schema=schema,
            using="default",
        )

        # 4. 创建 Field 的索引
        index_params = {
            'index_type': "IVF_FLAT",
            'metric_type': "L2",
            'params': {'nlist': 1}
        }
        collection.create_index(field_name="embedding", index_params=index_params)

        # 5. 挂载 Collection，变成可用状态
        collection.load()

    # ----- document -----
    @staticmethod
    def upsert_document(collection_name, entities):
        # 判断该集合是否存在
        if utility.has_collection(collection_name) is False:
            return False

        collection = Collection(
            name=collection_name,
            using="default",
        )
        collection.insert(entities)
        collection.flush()
        return True

    @staticmethod
    def search_document(collection_name, data, top_k):

        # 判断该集合是否存在
        if utility.has_collection(collection_name) is False:
            return False

        collection = Collection(
            name=collection_name,
            using="default",
        )

        documents = collection.search(
            data=data,
            anns_field="embedding",
            param={"metric_type": "L2", "params": {"nprobe": 1}, "offset": 0},
            limit=top_k,
            output_fields=["*"]
        )

        document_list = []
        for i in range(len(documents[0])):
            # 获取所有字段名
            fields = documents[0][i].entity.fields
            # 初始化最终的返回数据，并且添加 id 和 distance
            document = {'id': documents[0][i].id, 'distance': documents[0][i].distance}
            # 把其他字段也放进去最近返回数据
            for f in fields:
                document[f] = documents[0][i].entity.get(f)
            # 追加每一个文档
            document_list.append(document)
        return document_list
