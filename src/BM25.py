"""
BM25相似度计算
"""
from collections import Counter
from typing import Dict, List, Tuple
import jieba.posseg as pseg
from math import log


class BM25:
    """
    BM25相似度计算类
    """
    class DocumentInfo:
        """
        文档信息
        """
        def __init__(self, termFrequency: Counter, length: int, content: str):
            self.termFrequency: Counter = termFrequency  # 词 -> 词频
            self.length: int = length
            self.content: str = content

    K1 = 1.2
    b = 0.75
    K3 = 1
    SKIPPOSSET = {"c", "p", "u", "xc", "w"}  # 连词、介词、助词、其他虚词、标点

    def __init__(self):
        self.documentNumber: int = 0
        self.documentLengthSum: int = 0
        self.documentFrequency: Dict[str, int] = dict()  # 词 -> 包含词的文档数
        self.documentInfo: Dict[str, BM25.DocumentInfo] = dict()  # 文档id -> 文档信息

    @property
    def averageDocumentLength(self) -> float:
        """
        文档的平均长度
        """
        return self.documentLengthSum / self.documentNumber

    def __tokenize(self, text: str) -> List[str]:  # [item, item, item, ...]
        """
        文本分词，过滤无关词性，返回词项列表
        """
        cutResult = pseg.cut(text)
        cutResult = filter(lambda x: x.flag not in self.SKIPPOSSET, cutResult)  # 过滤无关词性
        termList = list(map(lambda x: x.word, cutResult))
        return termList

    def addDocument(self, documentID: str, documentContent: str) -> None:
        """
        添加文档
        """
        termList = self.__tokenize(documentContent)
        termFrequency = Counter(termList)
        # {item: frequency, item: frequency, ...}
        length = len(termList)

        self.documentNumber += 1
        self.documentLengthSum += length
        for term in termFrequency:
            if term not in self.documentFrequency:
                self.documentFrequency[term] = 0
            self.documentFrequency[term] += 1
        self.documentInfo[documentID] = BM25.DocumentInfo(termFrequency, length, documentContent)

    def query(self, text: str, limit: int) -> List[Tuple[str, str, float]]:
        """
        查询
        :param text: 查询文本
        :param limit: 返回数量限制
        :return 列表，元素为(documentID, documentContent, BM25Score)
        """
        queryTermList = self.__tokenize(text)
        queryTermFrequency = Counter(queryTermList)
        # {item: frequency, item: frequency, ...}

        documentScoreList: List[Tuple[str, float]] = list()  # [(documentID, BM25Score), (documentID, BM25Score), ...]
        for documentID in self.documentInfo:
            currentDocumentInfo = self.documentInfo[documentID]
            score = 0.0
            for queryItem in queryTermFrequency:
                if queryItem not in self.documentFrequency:
                    # 所有文档里都没有出现过的item是没有价值的
                    continue
                tf = currentDocumentInfo.termFrequency[queryItem] if queryItem in currentDocumentInfo.termFrequency\
                    else 0
                scorePart1 = log(1 + (self.documentNumber - self.documentFrequency[queryItem] + 0.5) /
                                 (self.documentFrequency[queryItem] + 0.5))
                scorePart2 = (self.K1 + 1) * tf / (self.K1 * ((1 - self.b) + self.b * currentDocumentInfo.length /
                                                              self.averageDocumentLength) + tf)
                scorePart3 = (self.K3 + 1) * queryTermFrequency[queryItem] / (self.K3 + queryTermFrequency[queryItem])
                score += scorePart1 * scorePart2 * scorePart3
            documentScoreList.append((documentID, score))

        documentScoreList.sort(key=lambda x: x[1], reverse=True)  # 按BM25分数倒序
        result = [(documentID, self.documentInfo[documentID].content, score)
                  for documentID, score in documentScoreList[:limit]]
        return result
