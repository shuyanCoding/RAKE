import os
import json
import time

from rake_nltk import Rake
from nltk.corpus import stopwords

import nltk
nltk.download('punkt')
nltk.download('stopwords')

# 定义一个函数来执行文本预处理
def preprocess_text(text):
    # 这里使用一个简单的文本预处理，去除换行符和多余空格
    text = text.replace('\n', ' ').replace('\r', ' ')
    text = ' '.join(text.split())
    return text

# 定义Enron数据集路径
enron_data_path = './maildir'

# 初始化 Rake
r = Rake(stopwords=stopwords.words('english'))

# 指定本地keywords-document-index的json文件路径
keywords_dict_path = 'keywords-document-index.json'

# # 从文件加载字典
# with open(keywords_dict_path, 'r') as json_file:
#     keywords_dict = json.load(json_file)

# 检查文件是否存在且非空
if os.path.exists(keywords_dict_path) and os.path.getsize(keywords_dict_path) > 0:
    # 从文件加载字典
    with open(keywords_dict_path, 'r') as json_file:
        keywords_dict = json.load(json_file)
else:
    # 如果文件不存在或为空，初始化一个空字典
    keywords_dict = {}

# 记录开始时间
start_time = time.time()

count = 1
# 遍历Enron数据集
for root, dirs, files in os.walk(enron_data_path):
    for file in files:
        file_path = os.path.join(root, file)

        # 解析邮件文件
        with open(file_path, 'r', encoding='latin1') as f:
            content = f.read()
            # 进行文本预处理
            preprocessed_content = preprocess_text(content)

            # 提取关键字
            r.extract_keywords_from_text(preprocessed_content)

            # 获取关键字和它们的分数
            keyword_scores = r.get_word_degrees()

            # # 获取前N个关键字
            # top_keywords = sorted(keyword_scores, key=keyword_scores.get, reverse=True)[:10]

            # 将关键字和文档索引保存在keywords-document-index.json字典中
            for keyword_score in keyword_scores:
                if keyword_score in keywords_dict:
                    keywords_dict[keyword_score].append(count)
                else:
                    keywords_dict[keyword_score] = [count]

                print(file_path, keyword_score, count)
        count = count + 1

# 记录结束时间
end_time = time.time()

# 计算并打印运行时间
elapsed_time = end_time - start_time
print(f"运行时间: {elapsed_time} 秒")

# 将字典写入文件
with open(keywords_dict_path, 'w') as json_file:
    json.dump(keywords_dict, json_file)
