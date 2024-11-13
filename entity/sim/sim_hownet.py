#!/usr/bin/env python3
# coding: utf-8
# File: sim.py
# Author: lhy<lhy_in_blcu@126.com,https://huangyong.github.io>
# Date: 18-4-27
import jieba.posseg as pseg

class SimHownet:
    def __init__(self):
        self.semantic_path = 'FollowUps\entity\sim\model\hownet.dat'
        self.semantic_dict = self.load_semanticwords()

    '''加载语义词典'''
    def load_semanticwords(self):
        semantic_dict = {}
        for line in open(self.semantic_path,encoding='utf-8'):
            words = [word for word in line.strip().replace(' ','>').replace('\t','>').split('>') if word !='']
            word = words[0]
            word_def = words[2]
            semantic_dict[word] = word_def.split(',')
        return semantic_dict

    '''基于语义计算语义相似度'''
    def calculate_semantic(self, DEF1, DEF2):
        DEF_INTERSECTION = set(DEF1).intersection(set(DEF2))
        DEF_UNION = set(DEF1).union(set(DEF2))
        return float(len(DEF_INTERSECTION))/float(len(DEF_UNION))

    '''比较两个词语之间的相似度'''
    def compute_similarity(self, word1, word2):
        DEFS_word1 = self.semantic_dict.get(word1, [])
        DEFS_word2 = self.semantic_dict.get(word2, [])
        scores = [self.calculate_semantic(DEF_word1, DEF_word2) for DEF_word1 in DEFS_word1 for DEF_word2 in DEFS_word2]
        if scores:
            return max(scores)
        else:
            return 0

    '''基于词语相似度计算句子相似度'''
    def distance(self, text1, text2):
        words1 = [word.word for word in pseg.cut(text1) if word.flag[0] not in ['u', 'x', 'w']]
        words2 = [word.word for word in pseg.cut(text2) if word.flag[0] not in ['u', 'x', 'w']]
        # print(words1, words2)
        score_words1 = []
        score_words2 = []
        if len(words1) > 0 and len(words2) > 0:
            for word1 in words1:
                score = max(self.compute_similarity(word1, word2) for word2 in words2)
                score_words1.append(score)
            for word2 in words2:
                score = max(self.compute_similarity(word2, word1) for word1 in words1)
                score_words2.append(score)
            similarity = max(sum(score_words1)/len(words1), sum(score_words2)/len(words2))
        else:
            similarity = 0
        return similarity

if __name__ == '__main__':
    simer = SimHownet()
    text1 = '南昌是江西的省会'
    text2 = '北京乃中国之首都'

    text1 = '的话'
    text2 = '还有什么打算买的智能设备吗？'

    sim = simer.distance(text1, text2)
    print(sim)

# test() [{'score': '0.669', 'word': '价格'}, {'score': '0.660', 'word': '旗舰店'}, {'score': '0.562', 'word': '可能'}, {'score': '0.556', 'word': '选择'}, {'score': '0.507', 'word': '一下'}, {'score': '0.506', 'word': '比较'}, {'score': '0.504', 'word': '如果'}, {'score': '0.503', 'word': '的话'}, {'score': '0.500', 'word': '淘宝'}]