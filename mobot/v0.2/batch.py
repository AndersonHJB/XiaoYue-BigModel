import util
import csv


# 这里需要改成知识库的文件所在路径
# 不需要带有列的表格
file_path = './data/knowledge_base.csv'

# 1. 读取 CSV 文件
with open(file_path, 'r') as file:
    reader = csv.reader(file)

    # 2. 把内容变成内存对象
    questions, answers, metadata = [], [], []
    for row in reader:
        questions.append(row[0])
        answers.append(row[1])
        metadata.append({"question": row[0], "answer": row[1]})

    print(f"CSV 文件读取到了 {len(metadata)} 行数据")

# 3. 把文本转换成向量
u = util.Util()
vectors = u.EmbeddingOpenAI.embed_documents(questions)

# 4. 数据处理成 CSV 格式（行格式）
ids = list(range(1, len(questions)+1))

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

print(f"写入了 {len(vectors)} 条 Document 在 {save_file_path} 文件中")

# 6. 组装成 Pinecone 要求的格式
documents = []
for i in range(len(vectors)):
    documents.append(
        (
            str(ids[i]),  # ID
            vectors[i],   # 向量
            metadata[i]   # 元数据
        )
    )

# 7. 写入数据到 Pinecone（重复id的数据会覆盖）
index = u.VDBPinecone.get_pinecone_index("mobot")
response = index.upsert(
    vectors=documents
)

# 8. 核对下写入数据的情况
print(f"Pinecone 数据库写入了 {response['upserted_count']} 条 Document")
