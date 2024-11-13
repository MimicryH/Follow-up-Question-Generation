import pandas as pd
from FollowUps.back_channel.tokenizer import Tokenizer, load_chinese_base_vocab
import torch
from FollowUps.back_channel.utils import load_bert, load_model_params

vocab_path = "FollowUps/back_channel/state_dict/roberta_wwm_vocab.txt"  # roberta模型字典的位置
word2idx = load_chinese_base_vocab(vocab_path)

dict_bc = {
    '1a': '嗯。',  # '1a': '嗯/ok/嗯哼',
    '2a': '好的。',
    '2b': '喔。',  # '2b': '喔/嗷/昂/奥/这样呀',
    '2c': '了解。',  # '明白/了解/哦/好吧'
    '3a': '真棒！',  # '真棒/666'
    '3b': '不错！',  # '挺好/不错'
    '4a': '确实。',  # '确实/好吧/对/是'
    '4b': '可以。',
    '5a': '啊这……',
    '5b': '怎么会这样？',
    '5c': '嗯？',  # '嗯？/诶？/啊哈？'
    '5d': '真的么？',  # '是么？/真的么？'
    '5e': '什么？',
    '5f': '厉害！',
    '5g': '哈哈！',
    '6a': '然后呢？',  # '然后呢？/之后呢？'
    '6b': '除此以外呢？',  # '还有呢？/除此以外呢？'
    '6c': '具体说说……',
    '6d': '比如说？',  # '比如说？/举个例子？'
    '6e': '随便说说？',
    '6f': '为什么？'
}

dict_bc_type = {
    '1a': 'bc',  # '1a': '嗯/ok/嗯哼',
    '2a': 'bc',  # '2a': '好的',
    '2b': 'bc',  # '2b': '喔/嗷/昂/奥/这样呀',
    '2c': 'bc',  # '明白/了解/哦/好吧'
    '3a': 'bc',  # '真棒/666'
    '3b': 'bc',  # '挺好/不错'
    '4a': 'bc',  # '确实/好吧/对/是'
    '4b': 'bc',  # '4b': '可以'
    '5a': 'bc',
    '5b': 'bc',
    '5c': 'bc',  # '嗯？/诶？/啊哈？'
    '5d': 'bc',  # '是么？/真的么？'
    '5e': 'bc',
    '5f': 'bc',
    '5g': 'bc',
    '6a': '然后呢？',  # '然后呢？/之后呢？'
    '6b': '除此以外呢？',  # '还有呢？/除此以外呢？'
    '6c': '具体说说',
    '6d': '比如说？',  # '比如说？/举个例子？'
    '6e': '随便说说',
    '6f': '为什么？'
}

target = list(dict_bc.keys())

tokenizer = Tokenizer(word2idx)
# 判断是否有可用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device: " + str(device))
# 定义模型
bert_model = load_bert(word2idx, model_name="roberta", model_class="cls", target_size=len(target))

## 加载预训练的模型参数～
load_model_params(bert_model, "FollowUps/back_channel/model/bc_model_2770.bin")
# 将模型发送到计算设备(GPU或CPU)
bert_model.to(device)


def get_bcs_questions_bert(txt, model=bert_model):
    token_ids, token_type_ids = tokenizer.encode(txt)
    # token_ids = padding([token_ids], max_length=256, ).to(device)
    token_ids = torch.tensor(token_ids, device=device).view(1, -1)
    prediction = model(token_ids).tolist()[0]
    prediction = pd.Series(index=target, data=prediction).sort_values(ascending=False)
    bcs = []
    questions = []
    for idx, prob in prediction.iteritems():
        if dict_bc_type[idx] != 'bc':
            questions.append((idx, dict_bc[idx], prob))
        else:
            bcs.append((idx, dict_bc[idx], prob))
    return bcs, questions


def bert_distance(a, b, model=bert_model):
    token_ids_a, _ = tokenizer.encode(a)
    token_ids_a = torch.tensor(token_ids_a, device=device).view(1, -1)
    token_ids_b, _ = tokenizer.encode(b)
    token_ids_b = torch.tensor(token_ids_b, device=device).view(1, -1)
    enc_a, _ = model.bert(token_ids_a, output_all_encoded_layers=True)
    cls_a = enc_a[-1][:, 0].squeeze()
    enc_b, _ = model.bert(token_ids_b, output_all_encoded_layers=True)
    cls_b = enc_b[-1][:, 0].squeeze()
    return ((cls_a-cls_b)**2).sum().item()


