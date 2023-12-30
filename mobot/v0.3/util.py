import database
import model


class Util:
    def __init__(self):
        super(Util, self).__init__()

        # 替换成你想要的任意模型，插件化设计

        # Language Model
        # self.ChatQwen = model.QwenLocal()
        # self.ChatQwen = model.QwenOnline()
        self.ChatLlama2 = model.Llama2()

        # Embedding Model
        self.EmbeddingM3E = model.M3E()

        # Vector Database
        self.VDBMilvus = database.Milvus()
