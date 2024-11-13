import urllib.request
import urllib.parse
import base64
import json
import time
import hashlib
import jieba.posseg as posg
import pandas as pd

#接口地址
url ="http://ltpapi.xfyun.cn/v1/ke"
#开放平台应用ID
x_appid = "5fa507ac"
#开放平台应用接口秘钥
api_key = "e01b42eb832ae1c431049ad7b11c0424"

koi_path = "FollowUps/entity/entities_of_intrest.txt"
koi_words = []
with open(koi_path, encoding='utf-8') as f:
    for w in f.readlines():
        koi_words.append(w.strip("\n"))
f.close()


def get_entity_from_xf(text):
    # print(text)
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
    j = json.loads(result.decode('utf-8'))
    print(j)
    if "ke" not in j["data"].keys():
        return []
    return j["data"]["ke"]


def is_all_chinese(strs):
    for _char in strs:
        if not '\u4e00' <= _char <= '\u9fa5':
            return False
    return True


def get_entities(answer):
    # print("answer"+answer)
    # print("Question"+Question)
    # print("Q"+Q)
    entities = get_entity_from_xf(answer)
    words = posg.cut(answer, use_paddle=True)
    answer_pos = {}
    for word, pos in words:
        if pos not in ["q", "r", "p", "c", "u", "xc", "w", "uj"]:
            answer_pos[word] = pos
    words = []
    confidence = []
    for e in entities:
        if is_all_chinese(e['word']):
            words.append(e['word'])
            if 'score' in e.keys():
                c = float(e['score'])
            elif 'final' in e.keys():
                c = float(e['final'])
            else:
                c = 1.
            confidence.append(c)

    df = pd.DataFrame({"word": words, "confidence": confidence})
    df = df.sort_values(by="confidence", ascending=False)
    result = []
    for idx, row in df.iterrows():
        if row['word'] in answer_pos.keys():
            result.append((row['word'], answer_pos[row['word']], row['confidence']))
    # print(df)
    return result


pos_func = {"n": lambda a: ["具体哪个"+a+"?", "具体什么"+a+"?"],  # "谁的"+a+"?",
            "t": lambda a: [a+"吗?"],
            "nr": lambda a: ["具体哪位"+a+"?"],
            "ns": lambda a: ["在"+a+"吗?"],
            "v": lambda a: ["具体什么时候"+a+"?", "谁"+a+"?", "和谁"+a+"?", "在哪"+a+"?", "怎么"+a+"?"],
            "vd": lambda a: ["具体什么时候"+a+"?", "谁"+a+"?", "和谁"+a+"?", "在哪"+a+"?", "怎么"+a+"?"],
            "vn": lambda a: ["具体什么时候"+a+"?", "谁"+a+"?", "和谁"+a+"?", "在哪"+a+"?", "怎么"+a+"?"],
            "a": lambda a: ["有多"+a+"?"],
            "d": lambda a: ["有多"+a+"?"],
            "PER": lambda a: ["是"+a+"吗?"],
            "LOC": lambda a: ["在"+a+"吗?"],
            "ORG": lambda a: ["在"+a+"吗?"],
            "TIME": lambda a: [a+"吗?"]
            }

# 标签	含义	标签	含义	标签	含义	标签	含义
# n	普通名词	f	方位名词	s	处所名词	t	时间
# nr	人名	ns	地名	nt	机构名	nw	作品名
# nz	其他专名	v	普通动词	vd	动副词	vn	名动词
# a	形容词	ad	副形词	an	名形词	d	副词
# m	数量词	q	量词	r	代词	p	介词
# c	连词	u	助词	xc	其他虚词	w	标点符号
# PER	人名	LOC	地名	ORG	机构名	TIME	时间


def get_feedback(answer, Question, used_questions):
    general_feedbacks = ['然后呢？', '没了？', '还有呢？']
    entities_pos = get_entities(answer)
    print('entities_pos in text: ', entities_pos)
    questions_ett_pos_confidence = []
    for e in entities_pos:
        if e[0] in koi_words:
            if e[1] in pos_func:
                questions = pos_func[e[1]](e[0])
                for q in questions:
                    if q not in used_questions:
                        questions_ett_pos_confidence.append([q, e[0], e[1], e[2]])
            else:
                questions_ett_pos_confidence.append([e[0] + '?', e[0], e[1], e[2]])  # no matter what pos, 'xxx?' should work. Right?

    if len(questions_ett_pos_confidence) < 1:
        random_n = int(time.time()) % len(general_feedbacks)
        # return [general_feedbacks[random_n]], []
        return []
    else:
        return questions_ett_pos_confidence
