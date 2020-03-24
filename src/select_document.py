"""
选择文本
"""
from time import time

from BM25 import BM25
from tqdm import tqdm
import pickle as pkl

# contentDict = dict()
# with open("../data/NCPPolicies_context.csv") as contentFile:
#     for line in contentFile:
#         id, content = line.strip().split("\t", 1)
#         contentDict[id] = content
#
# with open("../data/content_dict.pkl", "wb") as contentDictFile:
#     pkl.dump(contentDict, contentDictFile, pkl.HIGHEST_PROTOCOL)

# trainDataset = list()
# with open("../data/NCPPolicies_train.csv") as trainFile:
#     trainFile.readline()
#     for line in trainFile:
#         qid, aid, question, answer = line.strip().split("\t", 3)
#         trainDataset.append((qid, aid, question, answer))
#
# with open("../data/train_dataset.pkl", "wb") as trainDatasetFile:
#     pkl.dump(trainDataset, trainDatasetFile, pkl.HIGHEST_PROTOCOL)

# with open("../data/content_dict.pkl", "rb") as contentDictFile:
#     contentDict = pkl.load(contentDictFile)
#
# bm25 = BM25()
# for id in tqdm(contentDict):
#     bm25.addDocument(id, contentDict[id])
#
# with open("../data/bm25.pkl", "wb") as bm25File:
#     pkl.dump(bm25, bm25File, pkl.HIGHEST_PROTOCOL)

with open("../data/bm25.pkl", "rb") as bm25File:
    bm25: BM25 = pkl.load(bm25File)

with open("../data/train_dataset.pkl", "rb") as trainDatasetFile:
    trainDataset: list = pkl.load(trainDatasetFile)

correctNum = 0
for index, item in enumerate(trainDataset):
    if index % 20 == 0 and index > 0:
        print(index, correctNum / index)
    if item[1] == bm25.query(text=item[2], limit=1)[0][0]:
        correctNum += 1

print(correctNum / len(trainDataset))
