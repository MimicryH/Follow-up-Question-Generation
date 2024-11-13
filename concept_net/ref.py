import random
import time
import urllib.request
import urllib.parse
import urllib3
import json
import hashlib
import base64
import json
import numpy as np
import pandas as pd
# from aip import AipNlp
from FollowUps.concept_net.langconv import *
# from interviewDS.CN_GENERATOR.sim_simhash import *
from FollowUps.concept_net.text2vec.similarity import Similarity
from torch.multiprocessing import TimeoutError, Pool, set_start_method, Queue
import torch.multiprocessing as mp
import torch
from pytorch_pretrained_bert import BertTokenizer, BertForMaskedLM
import math
import re

w2v_path = r'interviewDS\CN_GENERATOR\cache\light_Tencent_AILab_ChineseEmbedding.bin'
sim = Similarity(w2v_path=w2v_path)

url ="http://ltpapi.xfyun.cn/v1/ke"
x_appid = "5fa507ac"
api_key = "e01b42eb832ae1c431049ad7b11c0424"

# APP_ID = '23038516'
API_KEY = 'Agtd3WQOS8PMHqVuynHhV8kk'
SECRET_KEY = 'GyIgzVoFK5C3YTATVBpFLlAuGPWDIPFG'
# client = AipNlp(APP_ID, API_KEY, SECRET_KEY)

try:
    set_start_method('spawn')
except RuntimeError:
    pass
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_root = 'FollowUps/concept_net/cache/'

def load_model():
    _model = BertForMaskedLM.from_pretrained(model_root + r"bert-base-chinese.tar.gz").to(device)
    _model.eval()
    _tokenizer = BertTokenizer.from_pretrained(model_root + r'bert-base-chinese-vocab.txt')
    return _tokenizer, _model


tokenizer, bert_model = load_model()


def getlabel(word):
    import jieba.posseg as pseg
    res = pseg.cut(word)
    for (word,tag) in res:
        return tag


def getitems(obj):
    result=[]
    for item in obj['edges']:
        if 'language' in item['end'].keys() and 'language' in item['start'].keys():
            char=item['end']['language']
            lang=item['start']['language']
            if char=='zh' and lang=='zh':
                rel=item['rel']['label']
                end=item['end']['label']
                start=item['start']['label']
                result.append([start,rel,end])
    return result


def getentity(text):
    body = urllib.parse.urlencode({'text': text}).encode('utf-8')
    param = {"type": "dependent"}
    x_param = base64.b64encode(json.dumps(param).replace(' ', '').encode('utf-8'))
    x_time = str(int(time.time()))
    x_checksum = hashlib.md5(api_key.encode('utf-8') + str(x_time).encode('utf-8') + x_param).hexdigest()
    x_header = {'X-Appid': x_appid,
                'X-CurTime': x_time,
                'X-Param': x_param,
                'X-CheckSum': x_checksum}
    req = urllib.request.Request(url, body, x_header)
    result = urllib.request.urlopen(req)
    result = result.read()
    j=json.loads(result.decode('utf-8'))
    try:
        entityresult = sorted(j["data"]["ke"],key=lambda item:float(item['score']),reverse=True)
    except KeyError:
        entityresult = []
    return entityresult


def gettextSimilarity(texta,textb):
    return sim.get_score(texta, textb)


def getFluency_xsm(sentence):
    if len(sentence) < 2:
        return 1000
    tokenize_input = tokenizer.tokenize(sentence)
    tensor_input = torch.tensor([tokenizer.convert_tokens_to_ids(tokenize_input)], device="cuda")
    predictions = bert_model(tensor_input)
    loss_fct = torch.nn.CrossEntropyLoss()
    loss = loss_fct(predictions.squeeze(), tensor_input.squeeze()).data
    return math.exp(loss)


def get_fluency(sentence):
    if len(sentence) < 2:
        return 1000
    tokenize_input = tokenizer.tokenize(sentence)
    num_of_mask = math.ceil(len(tokenize_input) * 0.15)
    choices = list(range(len(tokenize_input)))
    random.shuffle(choices)
    masked_tokenize_input = tokenize_input.copy()
    for i in choices[:num_of_mask]:
        masked_tokenize_input[i] = '[MASK]'
    tensor_input = torch.tensor([tokenizer.convert_tokens_to_ids(tokenize_input)], device="cuda")
    masked_tensor_input = torch.tensor([tokenizer.convert_tokens_to_ids(masked_tokenize_input)], device="cuda")
    predictions = bert_model(masked_tensor_input)
    loss_fct = torch.nn.CrossEntropyLoss()
    loss = loss_fct(predictions.squeeze(), tensor_input.squeeze()).data
    return math.exp(1/loss)

# def trad2simp(sentence):
#     sentence = Converter('zh-hans').convert(sentence)
#     return sentence

# def simp2trad(sentence):
#     sentence = Converter('zh-hant').convert(sentence)
#     return sentence


def csv2json(csv_fn, json_fn, i=0):
    '''
    triple.csv->triple.json i=0
    template.csv->template.json i=1
    '''
    import csv
    import json
    f = open(csv_fn, 'r', encoding='UTF-8')
    csvreader = csv.reader(f)
    final_list = list(csvreader)
    del final_list[0]
    _dict = {}
    if i == 0:
        for item in final_list:
            if item[0] not in _dict.keys():
                _dict[item[0]] = [item[:6]]
            else:
                _dict[item[0]].append(item[:6])
            if item[3] not in _dict.keys():
                _dict[item[3]] = [item[:6]]
            else:
                _dict[item[3]].append(item[:6])
    elif i == 1:
        for item in final_list:
            if item[1] not in _dict.keys():
                _dict[item[1]] = [item]
            else:
                _dict[item[1]].append(item)

    f2 = open(json_fn, 'w', encoding='utf8')
    f2.write(json.dumps(_dict, ensure_ascii=False))
    f2.close()

def getTempPOS(template, pattern):
    temp_pos = None
    obj = re.compile(pattern).search(template)
    if obj:
        temp_pos = obj.group(1)
    return temp_pos


def getSentence(triple_path=r"triples.json",
                template_path=r"templates.json",
                save_path=r"sentence.json"):
    with open(triple_path,encoding='utf-8') as f:
        triples_ref = json.load(f)
    with open(template_path,encoding='utf-8') as f:
        templates_ref = json.load(f)
    res = {}  # pd.DataFrame(columns=['template','triple','sent','simi','flue','score'])
    checked_triple = []
    for item in triples_ref.values():
        for triple in item:
            if triple in checked_triple:
                continue
            else:
                checked_triple.append(triple)
            W1 = triple[0]
            POS1 = triple[1].split(' ')
            rela = triple[2]
            W2 = triple[3]
            POS2 = triple[4].split(' ')
            for template in templates_ref[rela]:
                sent = ''
                template = template[0]
                temp_pos1 = getTempPOS(template, r"<W1 (.*?)>")
                temp_pos2 = getTempPOS(template, r"<W2 (.*?)>")
                if temp_pos1 and temp_pos1 not in POS1:
                    pass
                elif temp_pos2 and temp_pos2 not in POS2:
                    pass
                else:
                    if W1 and W2 and re.match(r"(.*?)<W1(.*?)>(.*?)", template) and \
                            re.match(r"(.*?)<W2(.*?)>(.*?)", template):
                        sent = re.sub(r'<W1(.*?)>', W1, template)
                        sent = re.sub(r'<W2(.*?)>', W2, sent)
                    elif W1 == "" and not re.match(r"(.*?)<W1(.*?)>(.*?)", template):
                        sent = re.sub(r'<W2(.*?)>', W2, template)
                    elif W2 == "" and not re.match(r"(.*?)<W2(.*?)>(.*?)", template):
                        sent = re.sub(r'<W1(.*?)>', W1, template)
                    if 'W0' in template:
                        sent = template
                    triple_name = "-".join(triple)
                    if sent != "":
                        if 'W0' in template:
                            fluency = get_fluency(re.sub(r'<W0(.*?)>', W1, template))
                        else:
                            fluency = get_fluency(sent)
                        if triple_name not in res.keys():
                            res[triple_name] = [[template, triple, str(fluency), sent]]
                        else:
                            res[triple_name].append([template, triple, str(fluency), sent])

    f2 = open(save_path, 'w', encoding='utf8')
    f2.write(json.dumps(res, ensure_ascii=False))
    f2.close()


def inferring_triples(triple_path=r"cache/triple4169.json", save_path=r"cache/inferred.json"):
    with open(triple_path, encoding='utf-8') as f:
        triples_ref = json.load(f)
    # print(triples_ref.keys())
    # print(triples_ref.values())
    selected_relations = ["Synonym", "SimilarTo", "IsA"]
    selected_source = []
    for item in triples_ref.values():
        for triple in item:
            if triple[2] in selected_relations:
                selected_source.append(triple)

    TwoGenerationTriples = []
    original_tripples = []
    for selected_trip in selected_source:
        if selected_trip[2] != "IsA":  # selected_trip[2] is "SimilarTo" or "Synonym":
            W1 = selected_trip[0]
            W2 = selected_trip[3]
            for item in triples_ref.values():
                for triple in item:
                    newTrip = triple.copy()
                    if W1 in triple and W2 not in triple:
                        if W1 == triple[0]:
                            newTrip[0] = W2
                        elif W1 == triple[3]:
                            newTrip[3] = W2
                        if newTrip not in TwoGenerationTriples:
                            TwoGenerationTriples.append(newTrip)
                    elif W2 in triple and W1 not in triple:
                        if W2 == triple[0]:
                            newTrip[0] = W1
                        elif W2 == triple[3]:
                            newTrip[3] = W1
                        if newTrip not in TwoGenerationTriples:
                            TwoGenerationTriples.append(newTrip)

    for selected_trip in selected_source:
        W1 = selected_trip[0]
        W2 = selected_trip[3]
        for item in triples_ref.values():
            for triple in item:
                original_tripples.append(triple)
                if selected_trip[2] == "IsA":
                    if triple[2] not in selected_relations and W2 in triple and W1 not in triple:
                        newTrip = triple.copy()
                        if W2 == triple[0]:
                            newTrip[0] = W1
                        elif W2 == triple[3]:
                            newTrip[3] = W1
                        if newTrip not in TwoGenerationTriples:
                            TwoGenerationTriples.append(newTrip)

    Triples = []
    for triple in TwoGenerationTriples:
        if (triple not in Triples) and (triple not in selected_source) and (triple not in original_tripples):
            Triples.append(triple)
    data = {}
    for item in Triples:
        if item[0] not in data.keys():
            data[item[0]] = [item]
        else:
            data[item[0]].append(item)

        if item[3] not in data.keys():
            data[item[3]] = [item]
        else:
            data[item[3]].append(item)
    f2 = open(save_path, 'w', encoding='utf8')
    f2.write(json.dumps(data, ensure_ascii=False))
    f2.close()


if __name__ == '__main__':
    # csv2json("cache\original_ali_relation.csv - triples.csv", "cache/triple4169_new.json", 0)
    # csv2json("cache/original_ali_relation.csv - templates.csv", "cache/template4169_new.json", 1)
    getSentence(triple_path=r"cache/triple4169_new.json",
                template_path=r"cache/template4169_new.json",
                save_path=r"cache/sentence4169_new.json")
    # inferring_triples('cache/triple4169_new.json', 'cache/inferred.json')
