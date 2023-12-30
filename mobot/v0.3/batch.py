import util
import csv

# 这里需要改成知识库的文件所在路径
# 不需要带有列的表格
file_path = './data/knowledge_base.csv'

# 1. 加载
u = util.Util()

# 2. 读取 CSV 文件
with open(file_path, 'r') as file:
    reader = csv.reader(file)

    # 2. 把数据加载到内存对象中
    questions = []
    answers = []
    qa_pairs = []
    for row in reader:
        questions.append(row[0])
        answers.append(row[1])
        qa_pairs.append(f'{row[0]}\n\n{row[1]}')

    print(f"CSV 文件读取到了 {len(questions)} 行数据")

# 3. 把文本转换成向量
vectors = u.EmbeddingM3E.encode(qa_pairs)
vector_dim = len(vectors[0])

# 4. 数据处理成 CSV 格式（行格式）
ids = list(range(1, len(qa_pairs)+1))

data_list = []
for i in range(len(questions)):
    data_list.append([ids[i], questions[i], answers[i], vectors[i]])

# 5. 写入数据到 CSV
save_file_path = './data/knowledge_base_with_embed.csv'

with open(save_file_path, "w", newline="") as f:
    writer = csv.writer(f)
    header = ("id", "question", "answer", "embedding")
    writer.writerow(header)
    writer.writerows(data_list)

# 6. 核对下写入数据的情况
print(f"写入了 {len(vectors)} 条 Document 在 {save_file_path} 文件中")


# 7. 数据处理成 Milvus Entity 格式 (列格式)
entity = [ids, questions, answers, vectors]

# 8. 将数据写入 Milvus 向量数据库
collection_name = "mobot"
u.VDBMilvus.create_collection(collection_name, dim=vector_dim)
result = u.VDBMilvus.upsert_document(collection_name, entity)
print(f"写入 Milvus 的结果为: {result}")
