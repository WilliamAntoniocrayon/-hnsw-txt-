from ceshisqlite import search_db_data
from transformers import BertTokenizer
import hnswlib
import numpy as np
from itertools import islice
import sqlite3



# hnsw维度
dim = 128

# hnsw存储向量数量
num_elements = 10000

# hnsw index持久化地址
index_path= '../hnswIndex.bin'

# 读取问答库里的内容
Question = ['0']       # 临时保存所有问题
f = open('../ceshi.txt', 'r', encoding ='utf-8')
for line in islice(f,1,None):   # 跳过第一行数据
    label = line.split(':')[0]
    question = line.split(':')[1]
    answer = line.split(':')[2]
    Question.append(question)
# print(Question)   # 检验是否全部加入list中






#   加载预训练字典和分词方法
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

# 训练数据
bert_outPut = tokenizer(Question,add_special_tokens=True,max_length = 128,padding='max_length', truncation=True, return_tensors="pt")


# 创建hnsw空间
p = hnswlib.Index(space='l2', dim=dim)  # possible options are l2, cosine or ip
p.init_index(max_elements=num_elements, ef_construction=100, M=16)
            # 定义可以存储在结构中的最大元素
            # 定义构建时间/精读权衡
            # 定义图中的最大传出连接数
p.set_ef(50)
# 将数据插入hnsw空间
p.add_items(bert_outPut.input_ids)
            # data:N*dim
            # ids
            # num_threads 设置要使用的cpu线程数（-1表示使用默认值）
            # add_items调用线程安全，knn_query相反

# hnsw space持久化，存储
p.save_index(index_path)


# 查找处理流程
# 问题从这里输入，q为问题

while True:
    flag = 0
    print("请输入你的问题：")
    q = input()

    print('Question:',q)    # 用户的提问
    query1 = tokenizer(q,add_special_tokens=True,max_length = 128,padding='max_length', truncation=True, return_tensors="pt")
    # 相似度计算，返回最相似的，k为需要的返回的最近的向量数量
    labels, distances = p.knn_query(query1.input_ids, k=1)
    # print('labels:',int(labels))      # 强制转化labels数据
    decode = tokenizer.decode(p.get_items(labels)[0]).replace(" ", "")
    outPut = decode[5:decode.find('[SEP]')]
    print(f"label = {labels} , output = {outPut}")

    labels = int(labels)

    f = open('../ceshi.txt', 'r', encoding ='utf-8')
    for line in islice(f,1,None):
        if int(line.split(':')[0]) is labels:
            answer = line.split(':')[2]
            print('answer:',answer)


    if q == "退出":
        break

f.close()
