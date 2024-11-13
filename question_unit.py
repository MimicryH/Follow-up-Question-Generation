import pandas as pd
import os
import random

LOG_PATH = 'FollowUps/participant_progress_log/'

col_names = ['username','informativeness', 'sentiment', 'topic', 'original_topic_answer',
             'entity_h', 'entity_h_answer', '自然度eh', '相关度eh',
             'CN_h', 'CN_h_answer', '自然度ch', '相关度ch',
             'bert_h', 'bert_h_answer', '自然度bh', '相关度bh',
             'num_q_h',
             'entity', 'entity_answer', '自然度e', '相关度e',
             'CN', 'CN_answer', '自然度c', '相关度c',
             'bert', 'bert_answer', '自然度b', '相关度b',
             'num_q'
             ]

sentimental_questions = ['工作日上班下班的时候有哪些开心或者不开心的事情吗？',
                         '周末有什么事是让你开心或者不开心的吗？',
                         '你使用过哪些喜欢或者讨厌的智能家电产品吗？',
                         '使用智能家电产品的时候，有什么喜欢或者讨厌的经历呢？',
                         '你有什么喜欢或者想买的个人数码产品吗？',
                         '使用智能音箱或语音助手的时候，有什么喜欢或者讨厌的经历呢？'
                         ]

agenda_questions = ['说说你的日常吧，工作日下班回到家一般会做些什么呢？',
                    '工作日上班下班的时候有哪些开心或者不开心的事情吗？',
                    '周末的时候一般都做些什么呢？',
                    '你使用过哪些喜欢或者讨厌的智能家电产品吗？',
                    '通常你是如何使用智能家电产品呢？',
                    '你有什么喜欢或者想买的个人数码产品吗？',
                    '你是怎么使用智能音箱或者语音助手的呢？',
                    '使用智能音箱或语音助手的时候，有什么喜欢或者讨厌的经历呢？',
                    '在乘车或者开车的时候，如果有一个智能语音助手，你会让它帮你做什么呢？',
                    '你理想中的语音助手是什么样的？'
                    ]

agenda_requirements = ['简单回答', '详细回答']


def prepare_agenda_questions():
    q_list = []
    for q in agenda_questions:
        for r in agenda_requirements:
            q_list.append(q + '[' + r + ']')
    random.shuffle(q_list)
    return q_list


class QuestionUnits:
    def __init__(self, username):
        if username+'.xlsx' in os.listdir(LOG_PATH):
            self.data = pd.read_excel(LOG_PATH+username+'.xlsx')
            for i, row in self.data.iterrows():
                if pd.isna(row['original_topic_answer']) or row['original_topic_answer'] == '':
                    self.current_question = i
                    break
        else:
            self.create_log(username)
            self.current_question = 0
        self.username = username
        self.original_topic_answer = None
        self.record = pd.DataFrame(index=['<entity>', '<CN>', '<bert>', '<entity>_human', '<CN>_human', '<bert>_human'],
                                   columns=['follow_ups', 'follow_up_answers', 'naturalness', 'relevance'])

    def create_log(self, username):
        self.data = pd.DataFrame(columns=col_names)
        agenda_list = prepare_agenda_questions()
        self.data['topic'] = agenda_list
        for i, row in self.data.iterrows():
            if row['topic'][:-6] not in sentimental_questions:
                row['sentiment'] = 'low'
            else:
                row['sentiment'] = 'high'

            if '[简单回答]' in row['topic']:
                row['informativeness'] = '[简单回答]'
            else:
                row['informativeness'] = '[详细回答]'

            row['topic'] = row['topic'][:-6]
            row['username'] = username
        self.data.to_excel(LOG_PATH+username+'.xlsx', index=False)

    def save_to_file(self):
        self.add_unit()
        print('save_to_file')
        print(self.data)
        self.data.to_excel(LOG_PATH+self.username+'.xlsx', index=False)
        self.reset()

    def reset(self):
        self.original_topic_answer = None
        self.record = pd.DataFrame(index=['<entity>', '<CN>', '<bert>', '<entity>_human', '<CN>_human', '<bert>_human'],
                                   columns=['follow_ups', 'follow_up_answers', 'naturalness', 'relevance'])

    def add_unit(self):
        count_q = 0
        count_q_human = 0
        print(self.record)
        for i, q in self.record['follow_ups'].items():
            if not pd.isna(q) and q != '':
                if '_human' in i:
                    count_q_human += 1
                else:
                    count_q += 1

        self.data['num_q'].iloc[self.current_question] = count_q
        self.data['num_q_h'].iloc[self.current_question] = count_q_human
        self.data['entity'].iloc[self.current_question] = self.record['follow_ups']['<entity>']
        self.data['CN'].iloc[self.current_question] = self.record['follow_ups']['<CN>']
        self.data['bert'].iloc[self.current_question] = self.record['follow_ups']['<bert>']
        self.data['entity_answer'].iloc[self.current_question] = self.record['follow_up_answers']['<entity>']
        self.data['CN_answer'].iloc[self.current_question] = self.record['follow_up_answers']['<CN>']
        self.data['bert_answer'].iloc[self.current_question] = self.record['follow_up_answers']['<bert>']
        self.data['entity_h_answer'].iloc[self.current_question] = self.record['follow_up_answers']['<entity>_human']
        self.data['CN_h_answer'].iloc[self.current_question] = self.record['follow_up_answers']['<CN>_human']
        self.data['bert_h_answer'].iloc[self.current_question] = self.record['follow_up_answers']['<bert>_human']


    def load_follow_ups(self):
        self.data = pd.read_excel(LOG_PATH+self.username+'.xlsx')
        self.original_topic_answer = self.data['original_topic_answer'].iloc[self.current_question]

    def save_preceding_response(self, r_text):
        self.original_topic_answer = r_text
        self.data['original_topic_answer'].iloc[self.current_question] = r_text

        self.data.to_excel(LOG_PATH+self.username+'.xlsx', index=False)
