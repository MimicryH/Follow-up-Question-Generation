
import numpy as np

from bert_seq2seq.tokenizer import Tokenizer, load_chinese_base_vocab
import torch
from torch.utils.data import Dataset, DataLoader
from bert_seq2seq.utils import load_bert, load_model_params
import time
from tqdm import tqdm

vocab_path = "G:/bert_seq2seq\examples/state_dict/roberta_wwm_vocab.txt"  # roberta模型字典的位置
word2idx = load_chinese_base_vocab(vocab_path)
dict_bc = {
    '1a': '嗯/ok/嗯哼',
    '2a': '好的',
    '2b': '喔/嗷/昂/奥/这样呀',
    '2c': '明白/了解/哦/好吧',
    '3a': '真棒/666',
    '3b': '挺好/不错',
    '4a': '确实/好吧/对/是',
    '4b': '可以',
    '5a': '啊这',
    '5b': '怎会如此？',
    '5c': '嗯？/诶？/啊哈？',
    '5d': '是么？/真的么？',
    '5e': '什么？',
    '5f': '厉害',
    '5g': '哈哈哈',
    '6a': '然后呢？/之后呢？',
    '6b': '还有呢？/除此以外呢？',
    '6c': '具体说说',
    '6d': '比如说？/举个例子？',
    '6e': '随便说说',
    '6f': '为什么？'
}

target = list(dict_bc.keys())


def get_tgt_tensor(_tgt):
    index_list = [target.index(t) for t in _tgt]
    _zeros = np.zeros(len(target))
    _zeros[index_list] = 1
    return _zeros


def read_corpus(data_path):
    """
    读原始数据
    """
    _src = []
    _tgt = []

    with open(data_path, encoding='utf-8') as f:
        lines = f.readlines()
    for line in lines:
        line = line[:-1].split("\t")
        _tgt.append(get_tgt_tensor(line[1:]))
        _src.append(line[0])
    return _src, _tgt


## 自定义dataset
class NLUDataset(Dataset):
    """
    针对特定数据集，定义一个相关的取数据的方式
    """
    def __init__(self, sents_src, sents_tgt) :
        ## 一般init函数是加载所有数据
        super(NLUDataset, self).__init__()
        # 读原始数据
        # self.sents_src, self.sents_tgt = read_corpus(poem_corpus_dir)
        self.sents_src = sents_src
        self.sents_tgt = sents_tgt

        self.idx2word = {k: v for v, k in word2idx.items()}
        self.tokenizer = Tokenizer(word2idx)

    def __getitem__(self, i):
        ## 得到单个数据
        # print(i)
        src = self.sents_src[i]
        tgt = self.sents_tgt[i]
        token_ids, token_type_ids = self.tokenizer.encode(src)
        output = {
            "token_ids": token_ids,
            "token_type_ids": token_type_ids,
            "target_id": tgt
        }
        return output

    def __len__(self):
        return len(self.sents_src)


def collate_fn(batch, max_len=256):
    """
    动态padding， batch为一部分sample
    """

    def padding(indice, max_length, pad_idx=0):
        """
        pad 函数
        """
        pad_indice = [[pad_idx] * max(0, max_length - len(item)) + item for item in indice]
        return torch.tensor(pad_indice)

    token_ids = [data["token_ids"] for data in batch]
    max_length = max([len(t) for t in token_ids])
    token_type_ids = [data["token_type_ids"] for data in batch]
    target_ids = [data["target_id"] for data in batch]
    target_ids = torch.tensor(target_ids, dtype=torch.float)

    token_ids_padded = padding(token_ids, max_length)
    token_type_ids_padded = padding(token_type_ids, max_length)
    # target_ids_padded = token_ids_padded[:, 1:].contiguous()
    if max_length > max_len:
        token_ids_padded = token_ids_padded[:, -max_len:]
        token_type_ids_padded = token_type_ids_padded[:, -max_len:]

    return token_ids_padded, token_type_ids_padded, target_ids


class Trainer:
    def __init__(self, data_path, model_name, model_path, batch_size=8, lr=1e-5):
        # 加载数据
        self.sents_src, self.sents_tgt = read_corpus(data_path)
        self.tokenier = Tokenizer(word2idx)
        # 判断是否有可用GPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("device: " + str(self.device))
        # 定义模型
        self.bert_model = load_bert(word2idx, model_name=model_name, model_class="cls", target_size=len(target))
        ## 加载预训练的模型参数～
        load_model_params(self.bert_model, model_path)
        # 将模型发送到计算设备(GPU或CPU)
        self.bert_model.to(self.device)
        # 声明需要优化的参数
        self.optim_parameters = list(self.bert_model.parameters())
        self.optimizer = torch.optim.Adam(self.optim_parameters, lr=lr, weight_decay=1e-3)
        # 声明自定义的数据加载器
        dataset = NLUDataset(self.sents_src, self.sents_tgt)
        self.dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    def train(self, _epoch, model_save_path):
        # 一个epoch的训练
        self.bert_model.train()
        return self.iteration(_epoch, dataloader=self.dataloader, model_save_path=model_save_path, train=True)

    def save(self, save_path):
        """
        保存模型
        """
        torch.save(self.bert_model.state_dict(), save_path)
        print("{} saved!".format(save_path))

    def iteration(self, _epoch, dataloader, model_save_path, train=True):
        total_loss = 0
        start_time = time.time() ## 得到当前时间
        step = 0
        for token_ids, token_type_ids, target_ids in tqdm(dataloader, position=0, leave=True):
            step += 1
            # if step % 2000 == 0:
            #     self.bert_model.eval()
            #     test_data = ["编剧梁馨月讨稿酬六六何念助阵 公司称协商解决", "西班牙BBVA第三季度净利降至15.7亿美元", "基金巨亏30亿 欲打开云天系跌停自救"]
            #     for text in test_data:
            #         text, text_ids = self.tokenier.encode(text)
            #         text = torch.tensor(text, device=self.device).view(1, -1)
            #         print(target[torch.argmax(self.bert_model(text)).item()])
            #     self.bert_model.train()

            token_ids = token_ids.to(self.device)
            # token_type_ids = token_type_ids.to(self.device)
            target_ids = target_ids.to(self.device)
            # 因为传入了target标签，因此会计算loss并且返回
            predictions, loss = self.bert_model(token_ids, labels=target_ids,)
            # 反向传播
            if train:
                # 清空之前的梯度
                self.optimizer.zero_grad()
                # 反向传播, 获取新的梯度
                loss.backward()
                # 用获取的梯度更新模型参数
                self.optimizer.step()

            # 为计算当前epoch的平均loss
            total_loss += loss.item()

        end_time = time.time()
        spend_time = end_time - start_time
        # 打印训练信息
        print("epoch is " + str(_epoch)+". loss is " + str(total_loss) + ". spend time is " + str(spend_time))
        # 保存模型
        self.save(model_save_path)
        return total_loss


trainer = Trainer(data_path="G:/bert_seq2seq\examples\data/data_ali_id_out_new.txt", model_name="roberta",
                  model_path="G:/bert_seq2seq\examples/state_dict/roberta_wwm_pytorch_model.bin")
train_epoches = 64
history = []
for epoch in range(train_epoches):
    # 训练一个epoch
    history.append(trainer.train(epoch, 'G:/bert_seq2seq\examples/bc_model_64.bin'))

from matplotlib import pyplot as plt
x = np.arange(train_epoches)
y = history
plt.title("Training Loss")
plt.xlabel("x axis caption")
plt.ylabel("y axis caption")
plt.plot(x, y)
plt.show()