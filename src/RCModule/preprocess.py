"""数据预处理"""
import os
import re
from typing import List

import pandas as pd

# pandas设置
pd.set_option('display.max_columns', None)
pd.set_option('max_colwidth', 25)
pd.set_option('display.width', None)

# 常量
RAW_DATA_DIR = "../../data"
DATA_DIR = "data"

if __name__ == "__main__":
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    # 处理文档信息
    with open(os.path.join(RAW_DATA_DIR, "NCPPolicies_context.csv")) as raw_content_file:
        with open(os.path.join(DATA_DIR, "context.csv"), "w") as content_file:
            for line in raw_content_file:
                id, content = line.strip().split("\t", 1)
                content = re.sub(r"\s+", "", content)
                content_file.write(f"{id}\t{content}\n")

    # 处理训练集信息
    context_df = pd.read_csv(os.path.join(DATA_DIR, "context.csv"), sep="\t")
    context_df.set_index("docid", drop=True, inplace=True)
    train_df = pd.read_csv(os.path.join(RAW_DATA_DIR, "NCPPolicies_train.csv"), sep='\t')
    train_df["question"] = train_df["question"].map(lambda s: re.sub(r"\s+", "", s))
    train_df["answer"] = train_df["answer"].map(lambda s: re.sub(r"\s+", "", s))
    start_pos_list: List[int] = []
    end_pos_list: List[int] = []
    not_match_index_list: List[int] = []
    for index, item in enumerate(train_df.itertuples()):
        text = context_df.at[item.docid, "text"]
        start_pos = text.find(item.answer)
        # 如果找不到（经过查看确实是训练集中的无效数据）
        if start_pos == -1:
            not_match_index_list.append(index)
        start_pos_list.append(start_pos)
        end_pos_list.append(start_pos + len(item.answer))

    train_df['start_pos'] = start_pos_list
    train_df['end_pos'] = end_pos_list
    train_df.drop(not_match_index_list, inplace=True)
    train_df.to_csv(os.path.join(DATA_DIR, "train.csv"), sep="\t", index=False, header=True)

    # 处理测试集信息
    test_df = pd.read_csv(os.path.join(RAW_DATA_DIR, "NCPPolicies_test.csv"), sep="\t")
    test_df["question"] = test_df["question"].map(lambda s: re.sub(r"\s+", "", s))
    test_df.to_csv(os.path.join(DATA_DIR, "test.csv"), sep="\t", index=False, header=True)
