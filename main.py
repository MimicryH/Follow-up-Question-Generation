import pandas as pd
from transitions import Machine
from basicDemo.models import *  # Messages,NextQLog,ConceptNetLog,IntentionLog
import FollowUps.follow_ups
import random
from FollowUps.question_unit import QuestionUnits


class State(object):
    states = ['introducing', 'confirming_introduction', 'agenda_asked', 'preparing', 'end', 'follow_up_asked']
    transitions = [
        {'trigger': 'introduced', 'source': 'introducing', 'dest': 'confirming_introduction'},
        {'trigger': 'confirmed_intro', 'source': 'confirming_introduction', 'dest': 'agenda_asked'},
        {'trigger': 'agenda_answered', 'source': 'agenda_asked', 'dest': 'preparing'},
        {'trigger': 'prepared', 'source': 'preparing', 'dest': 'follow_up_asked'},
        {'trigger': 'next_agenda', 'source': 'follow_up_asked', 'dest': 'agenda_asked'},
        {'trigger': 'next_agenda', 'source': 'agenda_asked', 'dest': 'agenda_asked'},
        {'trigger': 'agenda_finished', 'source': 'follow_up_asked', 'dest': 'end'},
        # {'trigger': 'neg', 'source': '1', 'dest': '12'},
    ]

    def __init__(self):
        self.machine = Machine(model=self, states=State.states, transitions=State.transitions, initial='introducing')


class MessageHandler(object):
    def __init__(self):
        self.state_machine = State()
        self.openId = 'unknown'
        self.username = 'unknown'
        self.fu = FollowUps.follow_ups.FollowUps()
        self.random_list = ['<entity>', '<CN>', '<bert>', '<entity>_human', '<CN>_human', '<bert>_human']
        self.follow_up_list = []
        self.current_follow_up = 0
        self.entity_df = None
        self.cn_df = None
        self.bert_questions = None
        self.pre_fu = None
        self.pre_fu_type = None
        self.log = None
        # self.next_q_classifier = next_q

    def save_msg(self, openid, msg_text, msg_type):
        msg_q = Messages(openId=openid, username=self.username, speaker='bot', text=msg_text, message_type=msg_type)
        if msg_type == '<CN>':
            # self.fu.update_used_triples(self.cn_df['triple'][0], self.cn_df['entity'][0])
            concept_net_log = ConceptNetLog(triples=self.cn_df['triple'],
                                            original_response=self.log.original_topic_answer,
                                            selected_triple=self.cn_df['triple'][0],
                                            template_scores=self.cn_df.to_json(orient='records', force_ascii=False))
            concept_net_log.save()
            msg_q.cn_log = concept_net_log
        elif msg_type == '<entity>':
            # self.fu.used_word_clarification_questions.append(followupQ)
            entity_log = EntityLog(original_response=self.log.original_topic_answer,
                                   entity_pos=self.entity_df.to_json(orient='records', force_ascii=False))
            entity_log.save()
            msg_q.entity_log = entity_log
        elif msg_type == '<bert>':
            # if self.fu.used_short_questions.qsize() == self.fu.used_short_questions.maxsize:
            #     self.fu.used_short_questions.get()
            #     print('self.fu.used_short_questions.put', self.bert_questions)
            #     self.fu.used_short_questions.put(self.bert_questions)
            bert_log = BertLog(original_response=self.log.original_topic_answer,
                               question_list='\n'.join(self.bert_questions))
            bert_log.save()
            msg_q.bert_log = bert_log
        msg_q.save()

    def is_question_finished(self, question):
        if len(self.log.data) > 0:
            for i, q in self.log.data['topic'].iteritems():
                print('q = ', q)
                if question in q:
                    return True
        return False

    def get_agenda_q(self):
        q_row = None
        for i in range(self.log.current_question, len(self.log.data)):
            if pd.isna(self.log.data['original_topic_answer'].iloc[i]) \
                    or self.log.data['original_topic_answer'].iloc[i] == '':
                q_row = self.log.data.iloc[i]
                self.log.current_question = i
                self.state_machine.next_agenda()
                break

        if q_row is None:
            response = '实验已完成哦。感谢！'
            r_type = 'end'
            self.state_machine.agenda_finished()
        else:
            response = ('话题 - {}/{}\n' \
                       + 'Q: ' +  q_row['topic'] + q_row['informativeness']).format(self.log.current_question + 1,
                                                                                    len(self.log.data))
            r_type = 'agenda'
        return response, r_type

    def init_follow_ups(self):
        self.log.load_follow_ups()
        self.fu.reset_the_dialog_cache()
        topic = self.log.data.iloc[self.log.current_question]['topic']
        self.entity_df = self.fu.gen_fu_entity(self.log.original_topic_answer, topic)
        if len(self.entity_df) > 0:
            for i, q in self.entity_df['question'].iteritems():
                if q not in self.log.data['entity'].values:
                    followupQ_entity = q
                    break
        else:
            followupQ_entity = ""
        # ConceptNet. associated info seeking.
        self.cn_df = self.fu.gen_fu_cn_df_no_repeat(self.log.original_topic_answer, topic, similarity_weight=1.0)
        followupQ_cn = ""
        if len(self.cn_df) > 0:  # if the CN works
            print(self.cn_df[:5])
            for i, q in self.cn_df['question'].iteritems():
                if q not in self.log.data['CN'].values:
                    followupQ_cn = q
                    break
        # bert. additional info seeking.
        _, self.bert_questions = self.fu.gen_back_channel(self.log.original_topic_answer)
        tempt_questions = {
            '<entity>': followupQ_entity,
            '<CN>': followupQ_cn,
            '<bert>': self.bert_questions[0],
            '<entity>_human': self.log.data['entity_h'].iloc[self.log.current_question],
            '<CN>_human': self.log.data['CN_h'].iloc[self.log.current_question],
            '<bert>_human': self.log.data['bert_h'].iloc[self.log.current_question]
        }
        # randomize
        random.shuffle(self.random_list)
        question_and_type = []
        for q_type in self.random_list:
            if tempt_questions[q_type] != "" and not pd.isna(tempt_questions[q_type]):
                question_and_type.append([tempt_questions[q_type], q_type])
        self.follow_up_list = question_and_type
        self.current_follow_up = 0
        self.log.record['follow_ups'] = tempt_questions.values()
        response = 'Q:' + self.log.data.iloc[self.log.current_question]['topic'] + '\n' \
                   + 'A: ' + self.log.original_topic_answer + '\n' \
                   + '\n-----------------------------------------\n' \
                   + 'Q:' + question_and_type[0][0] + '[正常回答]\n（问题无关或无法理解，可回复00跳过）'
        self.pre_fu_type = question_and_type[0][1]
        return response, question_and_type[0][1]

    def next_follow_up(self):
        self.current_follow_up += 1
        if self.current_follow_up > len(self.follow_up_list) - 1:
            return None, None
        else:
            self.pre_fu_type = self.follow_up_list[self.current_follow_up][1]
            response = 'Q:' + self.log.data.iloc[self.log.current_question]['topic'] + '\n' \
                       + 'A: ' + self.log.original_topic_answer + '\n' \
                       + '\n-----------------------------------------\n' \
                       + 'Q:' +  self.follow_up_list[self.current_follow_up][0] \
                       + '[正常回答]\n（问题无关或无法理解，可回复00跳过）'
            return response, self.follow_up_list[self.current_follow_up][1]

    def run(self, openid=None, r_text=None):
        print("self.state_machine.state = ", self.state_machine.state)
        print("r_text = ", r_text)
        self.openId = openid
        if self.state_machine.state == 'introducing':
            response = '实验模拟的是一次关于智能家电使用和购买情况的调研。你需要按照一些要求回答一系列问题：\n' \
                       '[简单回答]--用一句话简单概括。\n' \
                       '[详细回答]--尽量完整的回答，提供各种细节。\n' \
                       '[正常回答]--根据问题回答即可。\n' \
                       '-----------------------------------------\n' \
                       '如有问题，请联系主试人员。如无问题，回复你的名字开始实验。'
            r_type = 'intro'
            self.state_machine.introduced()
        elif self.state_machine.state == 'confirming_introduction':
            self.state_machine.confirmed_intro()
            self.username = r_text
            self.log = QuestionUnits(self.username)
            response, r_type = self.get_agenda_q()
        elif self.state_machine.state == 'agenda_asked':
            if r_text == '00':
                self.log.current_question += 1
                response, r_type = self.get_agenda_q()
            else:
                self.log.save_preceding_response(r_text)
                self.state_machine.agenda_answered()
                response = 'Q:' + self.log.data.iloc[self.log.current_question]['topic'] + '\n' \
                           + 'A: ' + self.log.original_topic_answer \
                           + '\n-----------------------------------------\n' \
                           + '追问准备中……'
                r_type = 'preparing'
        elif self.state_machine.state == 'preparing':
            if r_text == '123':
                response, r_type = self.init_follow_ups()
                self.state_machine.prepared()
            else:
                response = 'Q:' + self.log.data.iloc[self.log.current_question]['topic'] + '\n' \
                           + 'A: ' + self.log.original_topic_answer \
                           + '\n-----------------------------------------\n' \
                           + '追问准备中……'
                r_type = 'preparing'
        elif self.state_machine.state == 'follow_up_asked':
            self.log.record['follow_up_answers'][self.pre_fu_type] = r_text
            response, r_type = self.next_follow_up()
            if not response:
                self.log.save_to_file()
                self.log.current_question += 1
                response, r_type = self.get_agenda_q()

        elif self.state_machine.state == 'end':
            response = '实验已完成，你不会还想再做一遍吧？'
            r_type = 'end'
        # Save the message into database
        self.save_msg(openid, response, r_type)

        print("response = ", response)
        print("r_type = ", r_type)
        return response, r_type


if __name__ == "__main__":
    MH = MessageHandler()
    res = MH.run(1)
    print(res)
    while True:
        ipt = input()
        res = MH.run(1, r_text=ipt)
        if res:
            print(res)
        else:
            break
