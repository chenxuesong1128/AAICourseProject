"""阅读理解模块入口"""

import argparse
import logging
import os
import pickle as pkl
import random
from math import ceil
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import torch
from rouge import Rouge
from torch.nn import Linear, CrossEntropyLoss
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import BertTokenizer, AdamW, BertModel, BertPreTrainedModel, BertConfig

# 定义常量
MODEL_PATH = "../../model/"
OUTPUT_DIR = "output/"
DATA_DIR = "data/"
TRAIN_FILE = os.path.join(DATA_DIR, "train.csv")
CONTEXT_FILE = os.path.join(DATA_DIR, "context.csv")
CONTEXT_DICT = os.path.join(DATA_DIR, "context.dict.pkl")
QUESTION_LIST = os.path.join(DATA_DIR, "question.list.pkl")
TRAIN_SLICE_LIST = os.path.join(DATA_DIR, "train_slice.list.pkl")
DEV_SLICE_LIST = os.path.join(DATA_DIR, "dev_slice.list.pkl")
TRAIN_DATASET = os.path.join(DATA_DIR, "train_dataset.pkl")
DEV_DATASET = os.path.join(DATA_DIR, "dev_dataset.pkl")

MAX_SEQ_LEN = 512
MAX_QUESTION_LEN = 100
LEARNING_RATE = 1e-5
ADAM_EPSILON = 1e-8
SEED = 2020
TRAIN_EPOCH = 10
EVAL_STEP = 512

# 设置GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 定义logger
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger(__name__)


# 定义阅读理解模型结果
class BertForRC(BertPreTrainedModel):
    """阅读理解模型"""

    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        self.answer_index = Linear(config.hidden_size, 2)
        # self.dropout = Dropout(config.hidden_dropout_prob)
        self.classifier = Linear(config.hidden_size, 2)
        self.init_weights()

    def forward(self, input_ids, attention_mask, token_type_ids):
        """前向传播"""
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

        sequence_output = outputs[0]  # (B, T, 768)
        pooled_output = outputs[1]  # (B, 768)

        # sequence_output = self.dropout(sequence_output)
        index_logits = self.answer_index(sequence_output)  # (B, T, 2)
        start_logits, end_logits = index_logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)  # (B, T)
        end_logits = end_logits.squeeze(-1)  # (B, T)

        # classification
        # pooled_output = self.dropout(pooled_output)
        classifier_logits = self.classifier(pooled_output)  # (B, 2)

        return start_logits, end_logits, classifier_logits


def parse_args():
    """添加命令行参数信息"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action='store_true', help="Whether to run training.")
    parser.add_argument("--test", action='store_true', help="Whether to run testing.")
    parser.add_argument("--train_batch_size", default=8, type=int, help="Batch size for training.")
    parser.add_argument("--eval_batch_size", default=8, type=int, help="Batch size per for evaluation.")
    return parser.parse_args()


def set_seed(seed: int):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # noinspection PyUnresolvedReferences
    torch.cuda.manual_seed_all(seed)


class ContextItem:
    """文本数据"""

    def __init__(self, docid, text, tokenizer: BertTokenizer):
        self.docid = docid
        self.text = text
        self.tokens = tokenizer.tokenize(text)
        token_to_char_indexes = []
        self.char_to_token_indexes = []
        chars = []
        for index, token in enumerate(self.tokens):
            token_to_char_indexes.append(len(chars))
            chars_in_token = list(token)
            for char in chars_in_token:
                self.char_to_token_indexes.append(index)
                chars.append(char)
        token_to_char_indexes.append(len(chars))
        self.char_to_token_indexes.append(len(self.tokens))


def read_context_items(context_file: str, tokenizer: BertTokenizer) -> Dict[str, ContextItem]:
    """读取Context信息并切分"""
    context_df = pd.read_csv(context_file, sep="\t")
    context_dict = {}
    for context in tqdm(context_df.itertuples(), total=len(context_df)):
        context_dict[context.docid] = ContextItem(context.docid, context.text, tokenizer)
    return context_dict


class QuestionItem:
    """问题数据"""

    def __init__(self, id, question, context: ContextItem, tokenizer: BertTokenizer, start_char_index, end_char_index):
        self.id = id
        self.question = question
        self.question_tokens = tokenizer.tokenize(question)[:MAX_QUESTION_LEN]
        self.context: ContextItem = context
        self.answer_start_index = context.char_to_token_indexes[start_char_index]  # 答案在context.tokens中的开始位置
        self.answer_end_index = context.char_to_token_indexes[end_char_index]  # 答案在context.tokens中的结束位置

    @property
    def answer(self):
        """获取答案文本"""
        return "".join(self.context.tokens[self.answer_start_index:self.answer_end_index])


def read_question_items(train_file, context_dict: Dict[str, ContextItem], tokenizer: BertTokenizer) -> List[
    QuestionItem]:
    """读取问题数据"""
    train_df = pd.read_csv(train_file, sep="\t")
    question_items = []
    for item in tqdm(train_df.itertuples(), total=len(train_df)):
        context = context_dict[item.docid]
        question_items.append(QuestionItem(item.id, item.question, context, tokenizer, item.start_pos, item.end_pos))
    return question_items


class SliceItem:
    """句子切分数据"""

    def __init__(self, question: QuestionItem, slice_start_index, slice_end_index):
        self.question: QuestionItem = question
        self.slice_start_index = slice_start_index  # slice在question.context.tokens里的开始位置
        self.slice_end_index = slice_end_index  # slice在question.context.tokens里的结束位置

    @property
    def answer_start_index(self):
        """表示切片中答案（或者部分）区间的开始位置，如果不包含答案则为-1"""
        if self.slice_start_index >= self.question.answer_end_index \
                or self.slice_end_index <= self.question.answer_start_index:
            # 完全不包含答案
            return -1
        if self.slice_start_index <= self.question.answer_start_index:
            # 完全包含答案头部
            return self.question.answer_start_index - self.slice_start_index + len(self.question.question_tokens) + 2
        # 从答案中间开始
        return len(self.question.question_tokens) + 2

    @property
    def answer_end_index(self):
        """表示切片中答案（或者部分）区间的结束位置，如果不包含答案则为-1"""
        if self.slice_start_index >= self.question.answer_end_index \
                or self.slice_end_index <= self.question.answer_start_index:
            # 完全不包含答案
            return -1
        if self.question.answer_end_index <= self.slice_end_index:
            # 完全包含答案尾部
            return self.question.answer_end_index - self.slice_start_index + len(self.question.question_tokens) + 2
        # 从答案中间结束
        return self.slice_end_index - self.slice_start_index + len(self.question.question_tokens) + 2

    @property
    def answer_label(self):
        """如果切片包含答案部分，则为1，否则，为0"""
        if self.slice_start_index >= self.question.answer_end_index \
                or self.slice_end_index <= self.question.answer_start_index:
            # 完全不包含答案
            return 0
        return 1

    @property
    def input_tokens(self) -> List[str]:
        """Bert输入tokens"""
        non_padding_tokens = ['[CLS]'] + self.question.question_tokens + ['[SEP]'] + \
                             self.question.context.tokens[self.slice_start_index:self.slice_end_index] + ['[SEP]']
        return non_padding_tokens + ['[PAD]'] * (MAX_SEQ_LEN - len(non_padding_tokens))

    def input_ids(self, tokenizer: BertTokenizer) -> np.ndarray:
        """Bert输入token ids"""
        return np.array(tokenizer.convert_tokens_to_ids(self.input_tokens), dtype=np.int)

    @property
    def token_type_ids(self) -> np.ndarray:
        """token类型，问题为0，文本为1，pad为0"""
        question_len = len(self.question.question_tokens) + 2
        slice_len = self.slice_end_index - self.slice_start_index + 1
        padding_len = MAX_SEQ_LEN - question_len - slice_len
        return np.concatenate([
            np.zeros(question_len, dtype=np.int),
            np.ones(slice_len, dtype=np.int),
            np.zeros(padding_len, dtype=np.int)
        ])

    @property
    def attention_mask(self) -> np.ndarray:
        """输入的attention mask，pad为0，其他为1"""
        non_padding_len = len(self.question.question_tokens) + 2 + self.slice_end_index - self.slice_start_index + 1
        return np.concatenate([
            np.ones(non_padding_len, dtype=np.int),
            np.zeros(MAX_SEQ_LEN - non_padding_len, dtype=np.int)
        ])


def create_sentence_items(questions: List[QuestionItem]) -> List[SliceItem]:
    """生成句子切分数据"""
    slice_item_list_list: List[List[SliceItem]] = []
    for question in questions:
        slice_item_list: List[SliceItem] = []
        max_slice_token_num = MAX_SEQ_LEN - len(question.question_tokens) - 3  # [CLS], [SEP], [SEP]

        # 按照句号切分
        sentence_list = []
        sentence_start_index = 0
        for index, token in enumerate(question.context.tokens):
            if token == '。' or index == len(question.context.tokens) - 1:
                sentence_end_index = index + 1
                sentence_list.append((
                    sentence_start_index,
                    sentence_end_index
                ))
                sentence_start_index = sentence_end_index

        # 合并断句/分割长句
        buffer_len = 0
        slice_start_index = 0
        sentence_index = 0
        while sentence_index < len(sentence_list):
            sentence_start_index, sentence_end_index = sentence_list[sentence_index]
            sentence_len = sentence_end_index - sentence_start_index
            if sentence_len + buffer_len <= max_slice_token_num:
                # 缓冲区还能容纳新句子
                if buffer_len == 0:
                    # 缓冲区第一个句子，更新开始位置
                    slice_start_index = sentence_start_index
                # 当前句子加入缓冲区
                buffer_len += sentence_len
                if sentence_index == len(sentence_list) - 1:
                    # 如果当前句子是最后一个句子，缓冲区结束
                    slice_end_index = sentence_end_index
                    slice_item_list.append(SliceItem(question, slice_start_index, slice_end_index))
                    buffer_len = 0
                sentence_index += 1
                continue
            if sentence_len + buffer_len > max_slice_token_num and buffer_len > 0:
                # 当前缓冲区由内容且放不下新句子，表示一个阅读理解模型输入的结束
                slice_end_index = sentence_start_index
                slice_item_list.append(SliceItem(question, slice_start_index, slice_end_index))
                buffer_len = 0
                continue
            if sentence_len > max_slice_token_num:
                # 当前缓冲区为空但依然放不下新句子，要拆分新句子
                slice_num = ceil(sentence_len / max_slice_token_num)
                slice_len = ceil(sentence_len / slice_num)
                for i in range(slice_num):
                    slice_start_index = sentence_start_index + i * slice_len
                    slice_end_index = min(slice_start_index + slice_len, sentence_end_index)
                    slice_item_list.append(SliceItem(question, slice_start_index, slice_end_index))
                buffer_len = 0
                sentence_index += 1
        slice_item_list_list.append(slice_item_list)

    # 作简要统计并扁平化
    slice_items: List[SliceItem] = []
    answer_num, answer_slice_num = 0, 0
    single_answer_num, single_slice_num = 0, 0
    multi_answer_num, multi_slice_num, multi_answer_slice_num = 0, 0, 0
    for slice_item_list in slice_item_list_list:
        answer_num += 1
        answer_slice_num += len(slice_item_list)
        cnt = 0
        for slice_item in slice_item_list:
            slice_items.append(slice_item)
            if slice_item.answer_label == 1:
                cnt += 1
        if cnt == 1:
            single_answer_num += 1
            single_slice_num += len(slice_item_list)
        else:
            multi_answer_num += 1
            multi_slice_num += len(slice_item_list)
            multi_answer_slice_num += cnt
    logger.info('*** 切分结果分析 ***')
    logger.info(f'*** 问题数量: {answer_num}, 文本切分数量: {answer_slice_num}')
    logger.info(f'*** 答案不跨切分边界的问题: {single_answer_num}, 文本切分数量: {single_slice_num}')
    logger.info(f'*** 答案跨切分边界的问题: {multi_answer_num}, 文本切分数量: {multi_slice_num}, '
                f'答案平均长度: {multi_answer_slice_num / multi_answer_num}')
    logger.info(f'*** 正样例数量：{single_answer_num + multi_answer_slice_num}, '
                f'正负比例：{(single_answer_num + multi_answer_slice_num) / (answer_slice_num - single_answer_num - multi_answer_slice_num)}')
    return slice_items


class RCDataset(Dataset):
    """阅读理解数据集"""

    def __init__(self, slices: List[SliceItem], tokenizer: BertTokenizer):
        self.slices = slices
        self.input_ids = np.stack([slice.input_ids(tokenizer) for slice in slices])
        self.attention_mask = np.stack([slice.attention_mask for slice in slices])
        self.token_type_ids = np.stack([slice.token_type_ids for slice in slices])
        self.start_lable = np.array([slice.answer_start_index for slice in slices])
        self.end_lable = np.array([slice.answer_end_index for slice in slices])
        self.answer_label = np.array([slice.answer_label for slice in slices])

    def __len__(self) -> int:
        return len(self.slices)

    def __getitem__(self, index):
        return (
            self.input_ids[index], self.attention_mask[index], self.token_type_ids[index],
            self.start_lable[index], self.end_lable[index], self.answer_label[index],
            index
        )

    def get_slices(self, indexes: torch.Tensor) -> List[SliceItem]:
        """获取指定序号的切分数据"""
        return [self.slices[i.item()] for i in indexes]


class Result:
    """计算预测结果的rouge-L"""

    def __init__(self):
        self.slices = {}
        self.results = {}
        self.best_scores = {}
        self.unknown_examples = {}

    @staticmethod
    def is_valid_index(slice: SliceItem, index: Tuple[int, int]) -> bool:
        """Return whether valid index or not.
        """
        start_index, end_index = index
        if end_index <= start_index:  # bert seq 中start&end相对位置正确
            return False
        text_start_index = len(slice.question.question_tokens) + 2
        text_end_index = text_start_index + slice.slice_end_index - slice.slice_start_index
        # bert seq中start不在question范围，也不在padding的范围
        if start_index < text_start_index + 2 or start_index >= text_end_index:
            return False
        # bert seq中end不在question范围，也不在padding的范围
        if end_index <= text_start_index + 2 or end_index > text_end_index:
            return False
        return True

    def update(self, slices: List[SliceItem], start_preds: torch.Tensor, end_preds: torch.Tensor,
               class_preds: torch.Tensor):
        """更新预测结果"""
        # 从分数top k的span中选择合适的span
        slice_num = len(slices)
        start_logits, start_index = torch.topk(start_preds, 10, dim=1)
        end_logits, end_index = torch.topk(end_preds, 10, dim=1)
        indices = [
            [
                (int(start), int(end) + 1) for start in start_index[i]  # 开区间
                for end in end_index[i]
            ]
            for i in range(slice_num)
        ]
        logits = torch.tensor([
            [
                (start_preds[i][indice[0]] + end_preds[i][indice[1] - 1]) if self.is_valid_index(slices[i],
                                                                                                 indice) else -1e3
                for indice in indices[i]
            ]
            for i in range(slice_num)
        ])
        logits, logit_indices = torch.max(logits, dim=1)
        final_indices = [indices[i][logit_indices[i].item()] for i in range(slice_num)]
        for i, slice in enumerate(slices):
            if slice.question.id not in self.best_scores.keys():
                self.best_scores[slice.question.id] = 0.0
            if self.best_scores[slice.question.id] < logits[i]:
                #   todo          and int(torch.max(class_preds,dim=1)[1][i])==0: # predict 'known'
                self.best_scores[slice.question.id] = logits[i]
                self.slices[slice.question.id] = slice
                self.results[slice.question.id] = [slice.slice_start_index, final_indices[i]]
            # 预测为无答案的样本
            if slice.question.id not in self.results.keys():
                self.unknown_examples[slice.question.id] = slice
            elif slice.question.id in self.unknown_examples.keys():
                self.unknown_examples.pop(slice.question.id)

    def generate_predictions(self):
        """Generate predictions of each examples.
        """
        answers = []
        logger.info("*** generate predictions ***")
        logger.info("*** eval examples: {} ***".format(len(self.best_scores)))
        logger.info("*** known examples: {} ***".format(len(self.results)))
        logger.info("*** unknown examples: {} ***".format(len(self.unknown_examples)))
        assert len(self.best_scores) == len(self.slices) + len(self.unknown_examples)
        for id in self.best_scores.keys():
            if id in self.results.keys() and id in self.slices.keys():
                doc_start, index = self.results[id]
                slice: SliceItem = self.slices[id]
                passage_token_start = doc_start + index[0] - len(slice.question.question_tokens) - 2
                passage_token_end = doc_start + index[1] - len(slice.question.question_tokens) - 2
                assert 0 <= passage_token_start < len(slice.question.context.tokens)
                assert 0 < passage_token_end <= len(slice.question.context.tokens)
                answer = "".join(slice.question.context.tokens[passage_token_start:passage_token_end])
            else:
                answer = '疫情'  # 该样本经过预测没有答案
                slice = self.unknown_examples[id]
            answers.append({'id': id, 'pred': answer, 'label': slice.question.answer})
        return answers

    def score(self):
        data = self.generate_predictions()
        id = [d['id'] for d in data]
        prediction = [d['pred'] for d in data]
        label = [d['label'] for d in data]
        df = pd.DataFrame({'id': id, 'prediction': prediction, 'label': label})
        hyps, refs = map(list, zip(*[[' '.join(list(d['pred'])), ' '.join(list(d['label']))] for d in data]))
        rouge = Rouge()
        scores = rouge.get_scores(refs, hyps, avg=True)
        return scores['rouge-l']['f'], df


def main():
    """程序入口"""

    logger.info("解析参数")
    args = parse_args()

    # 设置随机种子
    set_seed(SEED)

    # 建立输出文件夹
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    logger.info("加载bert配置")
    tokenizer: BertTokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
    config: BertConfig = BertConfig.from_pretrained(MODEL_PATH)

    logger.info("训练过程")
    if args.train:
        logger.info("加载bert模型")
        model = BertForRC.from_pretrained(MODEL_PATH, config=config).cuda()

        logger.info("加载训练集")
        if os.path.exists(TRAIN_DATASET) and os.path.exists(DEV_DATASET):
            with open(TRAIN_DATASET, "rb") as train_dataset_pkl:
                train_dataset: RCDataset = pkl.load(train_dataset_pkl)
            with open(DEV_DATASET, "rb") as dev_dataset_pkl:
                dev_dataset: RCDataset = pkl.load(dev_dataset_pkl)
        else:
            logger.info("读取Context数据")
            if os.path.exists(CONTEXT_DICT):
                with open(CONTEXT_DICT, "rb") as context_dict_pkl:
                    context_dict: Dict[str, ContextItem] = pkl.load(context_dict_pkl)
            else:
                context_dict = read_context_items(CONTEXT_FILE, tokenizer)
                with open(CONTEXT_DICT, "wb") as context_dict_pkl:
                    pkl.dump(context_dict, context_dict_pkl, pkl.HIGHEST_PROTOCOL)
            logger.info("读取Question数据")
            if os.path.exists(QUESTION_LIST):
                with open(QUESTION_LIST, "rb") as question_list_pkl:
                    question_list: List[QuestionItem] = pkl.load(question_list_pkl)
            else:
                question_list = read_question_items(TRAIN_FILE, context_dict, tokenizer)
                with open(QUESTION_LIST, "wb") as question_list_pkl:
                    pkl.dump(question_list, question_list_pkl, pkl.HIGHEST_PROTOCOL)
            logger.info("生成Slice数据")
            if os.path.exists(TRAIN_SLICE_LIST) and os.path.exists(DEV_SLICE_LIST):
                with open(TRAIN_SLICE_LIST, "rb") as train_slice_list_pkl:
                    train_slice_list: List[SliceItem] = pkl.load(train_slice_list_pkl)
                with open(DEV_SLICE_LIST, "rb") as dev_slice_list_pkl:
                    dev_slice_list: List[SliceItem] = pkl.load(dev_slice_list_pkl)
            else:
                train_slice_list = create_sentence_items(question_list[:-200])
                dev_slice_list = create_sentence_items(question_list[-200:])

                with open(TRAIN_SLICE_LIST, "wb") as train_slice_list_pkl:
                    pkl.dump(train_slice_list, train_slice_list_pkl, pkl.HIGHEST_PROTOCOL)
                with open(DEV_SLICE_LIST, "wb") as dev_slice_list_pkl:
                    pkl.dump(dev_slice_list, dev_slice_list_pkl, pkl.HIGHEST_PROTOCOL)
            train_dataset = RCDataset(train_slice_list, tokenizer)
            with open(TRAIN_DATASET, "wb") as train_dataset_pkl:
                pkl.dump(train_dataset, train_dataset_pkl, pkl.HIGHEST_PROTOCOL)
            dev_dataset = RCDataset(dev_slice_list, tokenizer)
            with open(DEV_DATASET, "wb") as dev_dataset_pkl:
                pkl.dump(dev_dataset, dev_dataset_pkl, pkl.HIGHEST_PROTOCOL)

        train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.train_batch_size)

        optimizer: AdamW = AdamW(model.parameters(), lr=LEARNING_RATE, eps=ADAM_EPSILON)

        logger.info("***** Running training *****")
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num epoch = %d", TRAIN_EPOCH)
        logger.info("  Learning rate = %f", LEARNING_RATE)

        model.train()

        best_acc = 0
        output_file = os.path.join(OUTPUT_DIR, "results.txt")
        with open(output_file, "w") as writer:
            writer.write('*' * 80 + '\n')
        for epoch in range(TRAIN_EPOCH):
            for index, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
                input_ids, attention_mask, token_type_ids, start_labels, end_labels, answer_labels, feature_indexs = batch
                input_ids = input_ids.cuda()
                attention_mask = attention_mask.cuda()
                token_type_ids = token_type_ids.cuda()
                start_labels = start_labels.cuda()
                end_labels = end_labels.cuda()
                answer_labels = answer_labels.cuda()
                start_logits, end_logits, classifier_logits = model(input_ids=input_ids, token_type_ids=token_type_ids,
                                                                    attention_mask=attention_mask)
                start_loss = CrossEntropyLoss(ignore_index=-1)(start_logits, start_labels)
                end_loss = CrossEntropyLoss(ignore_index=-1)(end_logits, end_labels)
                class_loss = CrossEntropyLoss()(classifier_logits, answer_labels)
                loss = start_loss + end_loss + 2 * class_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if (index + 1) % EVAL_STEP == 0 or (index + 1) == len(train_dataloader):
                    dev_dataloader = DataLoader(dev_dataset, batch_size=args.eval_batch_size)
                    model.eval()
                    with torch.no_grad():
                        result = Result()
                        for dev_batch in tqdm(dev_dataloader, leave=False):
                            dev_input_ids, dev_attention_mask, dev_token_type_ids, dev_start_labels, dev_end_labels, \
                            dev_answer_labels, dev_slice_indexs = dev_batch
                            dev_input_ids = dev_input_ids.cuda()
                            dev_attention_mask = dev_attention_mask.cuda()
                            dev_token_type_ids = dev_token_type_ids.cuda()
                            start_preds, end_preds, class_preds = model(input_ids=dev_input_ids,
                                                                        token_type_ids=dev_token_type_ids,
                                                                        attention_mask=dev_attention_mask)
                            start_preds = start_preds.detach().cpu()
                            end_preds = end_preds.detach().cpu()
                            class_preds = class_preds.detach().cpu()
                            slices = dev_dataset.get_slices(dev_slice_indexs)
                            result.update(slices, start_preds, end_preds, class_preds)

                    scores, predict_df = result.score()
                    model.train()
                    result = {'eval_accuracy': scores, 'loss': loss, "epoch": epoch, "batch": index,
                              "best_acc": best_acc}

                    with open(output_file, "a") as writer:
                        logger.info(str(result))
                        writer.write(str(result))
                        writer.write('*' * 80)
                        writer.write('\n')
                    if scores > best_acc:
                        logger.info("=" * 80)
                        logger.info("Best ACC %f" % scores)
                        logger.info("Saving Model......")
                        best_acc = scores
                        # save predict dataframe
                        predict_df.to_csv(os.path.join(OUTPUT_DIR, 'eval_prediction_text.csv'),
                                          index=False)
                        # Save a trained model
                        output_model_file = os.path.join(OUTPUT_DIR, "pytorch_model.bin")
                        torch.save(model.state_dict(), output_model_file)
                        logger.info("=" * 80)
                    else:
                        logger.info("=" * 80)
        with open(output_file, "a") as writer:
            writer.write('bert_acc: %f' % best_acc)


if __name__ == "__main__":
    main()
