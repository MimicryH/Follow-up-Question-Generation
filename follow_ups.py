import pandas as pd
from FollowUps.concept_net.CN import CN_generator
import FollowUps.entity.entity_fu as entity_fu
from FollowUps.back_channel import back_channel_prediction
from FollowUps.back_channel.model.roberta_model import BertEmbeddings, BertConfig
from FollowUps.back_channel.utils import load_model_params
from queue import Queue
import string
from zhon.hanzi import punctuation


def similarity_bert(a, b):
    return 1 / back_channel_prediction.bert_distance(a, b)


def without_punctuation(text):
    res = text
    for i in string.punctuation:
        res = res.replace(i, '')
    for i in punctuation:
        res = res.replace(i, '')
    return res


class FollowUps:
    def __init__(self):
        config = BertConfig(len(back_channel_prediction.word2idx))
        self.embeddings = load_model_params(BertEmbeddings(config).to('cuda'),
                                            "FollowUps/back_channel/state_dict/roberta_wwm_pytorch_model.bin")
        self.cn = CN_generator(triple_path=r"FollowUps/concept_net/cache/triple_inferred.json",
                               sentence_path=r"FollowUps/concept_net/cache/sentence_inferred.json")
        # self.how_net = entity_fu.hownet
        self.get_fluency = self.cn.get_fluency
        self.used_short_questions = Queue(maxsize=3)
        self.used_word_clarification_questions = []
        self.used_triples = []
        self.available_triple_entity = []
        self.bert_done = None

    def reset_the_dialog_cache(self):
        self.used_word_clarification_questions = []
        self.used_short_questions.queue.clear()
        self.used_triples = []
        self.available_triple_entity = []
        self.bert_done = None

    def gen_fu_entity(self, input_text, topic_question):
        questions_ett_pos_confidence = entity_fu.get_feedback(input_text, topic_question, self.used_word_clarification_questions)
        # print("before the repeatability check: ", questions_ett_pos)
        for q_e_p_c in questions_ett_pos_confidence:
            q_e_p_c.append(self.get_fluency(q_e_p_c[0]) + q_e_p_c[3])
        res_DF = pd.DataFrame(questions_ett_pos_confidence,
                              columns=['question', 'entity', 'pos',
                                       'confidence', 'score']).sort_values(by='score', ascending=False)
        print(res_DF[:3])
        return res_DF

    def gen_back_channel(self, input_text, top_n=5, only_bc=False):
        if not only_bc and self.bert_done is not None:
            question = self.bert_done
            self.bert_done = None
            if self.used_short_questions.qsize() == self.used_short_questions.maxsize:
                self.used_short_questions.get()
            self.used_short_questions.put(question)
            return None, question

        bcs, questions = back_channel_prediction.get_bcs_questions_bert(input_text)
        bcs = [bc[1] for bc in bcs]  # (idx, dict_bc[idx], prob)
        questions_no_repeat = []
        for q in questions:
            if q[1] not in list(self.used_short_questions.queue):
                questions_no_repeat.append(q[1])

        if len(questions_no_repeat) > 0:
            if len(questions_no_repeat) > top_n:
                questions_no_repeat = questions_no_repeat[:top_n]
            if only_bc:
                self.bert_done = questions_no_repeat[0]
                return bcs[0], None
            else:
                print('questions_no_repeate[:3]: ', questions_no_repeat[:3])
                return bcs[0], questions_no_repeat
        else:
            return bcs[0], None

    def gen_fu_cn_no_repeat(self, input_text):
        entities = self.cn.match_entities(input_text)
        triple_entitiy = self.cn.get_triples_entities(entities)
        print(triple_entitiy)
        # for t_e in triple_entitiy:
        #     if t_e[0] not in self.used_triples:
        #         self.available_triple_entity.append(t_e)
        # return self.cn.genSentences(input_text, self.available_triple_entity)
        return self.cn.genSentences(input_text, triple_entitiy)  #  ['template', 'triple', 'entity','question', 'flue']

    def gen_fu_cn_df_no_repeat(self, input_text, context, similarity_weight=1.0):
        res = self.gen_fu_cn_no_repeat(input_text)
        if len(res) > 0:
            similarity = []
            for _, row in res.iterrows():
                similarity.append(similarity_bert(without_punctuation(row['question']),
                                                  without_punctuation(input_text+context)))
            #  normalization
            similarity = [float(i) / max(similarity) for i in similarity]
            res['flue'] = [float(i) / max(res['flue']) for i in res['flue']]
            total = [similarity[i] * similarity_weight + res['flue'][i] for i in range(len(similarity))]
            res['similarity'] = similarity
            res['total'] = total
            res = res.sort_values("total", ascending=False)
        return res  #  ['template', 'triple', 'entity','question', 'flue', 'similarity', 'total']


    def gen_all_questions(self, input_text, topic_question):
        return {'entity': self.gen_fu_entity(input_text, topic_question),
                'cn': self.gen_fu_cn(input_text),
                'bc': self.gen_back_channel(input_text, 7)}

    def get_questions_with_scores(self, input_text, topic_question):
        qs = self.gen_all_questions(input_text, topic_question)
        q_list = []
        # similarity_hownet = []
        similarity_b = []
        fluency_bert = []
        for key in qs.keys():
            for q in qs[key]:
                if key == 'bc':
                    f = 10 * q[2]
                    q = q[1]
                else:
                    f = self.get_fluency(q)
                q_list.append(q)
                # ???
                # The similarity between generated question and the topic_question ?
                # OR between generated question and the topic_question + input_text ?
                # sh = self.similarity_hownet(q, topic_question)
                sb = similarity_bert(q, topic_question)

                # similarity_hownet.append(sh)
                similarity_b.append(sb)
                fluency_bert.append(f * len(q))
        # similarity_hownet = [float(i) / max(similarity_hownet) for i in similarity_hownet]
        similarity_b = [float(i) / max(similarity_b) for i in similarity_b]
        fluency_bert = [float(i) / max(fluency_bert) for i in fluency_bert]
        # total = [similarity_b[i] + fluency_bert[i] for i in range(len(similarity_hownet))]
        return q_list, similarity_b, fluency_bert

    def update_used_triples(self, triple, entity):
        if triple not in self.used_triples:
            self.used_triples.append(triple)
        if [triple, entity] in self.available_triple_entity:
            self.available_triple_entity.remove([triple, entity])


def save_test_result(fu, test_smpls):
    data = []
    for t in test_smpls:
        questions = fu.gen_all_questions(t[0], t[1])
        entity_questions = '\n'.join(questions['entity'])
        cn_questions = '\n'.join(questions['cn'])
        bc_q = []
        for bc in questions['bc']:
            bc_q.append(bc[1] + '-- %.2f' % bc[2])
        bc_q = '\n'.join(bc_q)
        count_q = len(questions['entity']) + len(questions['cn']) + len(questions['bc'])
        data.append([t[1], t[0], entity_questions, cn_questions, bc_q, count_q])
    df = pd.DataFrame(data, columns=["例行问题", "用户回答", "entity", "cn", "back channel", "count_q"])
    df.to_excel('test_result.xlsx')
    print(df)


def print_sorted_result(fu, test_smpls):
    threshold = .35
    topics = []
    input_text = []
    follow_ups = []
    s_bert = []
    fluency_bert = []
    for sample in test_smpls:
        q_list, s_b, f_b = fu.get_questions_with_scores(sample[0], sample[1])
        df = pd.DataFrame({"question": q_list, "similarity_bert": s_b,
                          "fluency_bert": f_b})
        df = df[(df['similarity_bert'] > threshold) & (df['fluency_bert'] > threshold)]
        if len(df) < 1:
            print(sample, " no question...")
        df = df.sort_values(by="total", ascending=False)
        count = 0
        for idx, row in df.iterrows():
            if count > 10:
                break
            count += 1
            topics.append(sample[1])
            input_text.append(sample[0])
            follow_ups.append(row['question'])
            s_bert.append(row['similarity_bert'])
            fluency_bert.append(row['fluency_bert'])

    df = pd.DataFrame({"例行问题": topics, "用户回答": input_text, "生成问题": follow_ups,
                       "相似度bert": s_bert, "流畅度": fluency_bert})
    df.to_excel('test_result_top_10_new.xlsx')


def test_excel(path):
    fu = FollowUps()
    df = pd.read_excel(path)
    test_smps = []
    for idx, row in df.iterrows():
        # print(row)
        test_smps.append((row['用户回答'], row['前文']))
    print_sorted_result(fu, test_smps)


if __name__ == '__main__':
    fu = FollowUps()
    print(fu.get_questions_with_scores('比如回到家的时候，让他开空调、开灯或者放音乐。', '智能应该是怎么理解的？'))
    # test_excel('测试集.xlsx')
    # print(fu.gen_fu_cn("智能应该是怎么理解的？这可能有时候不需要特意上手机上去。你就直接跟他一说，就可以了，他就可以帮助做到这方面。"))
    # from concept_net import ref
    # ref.getSentence(triple_path=r"concept_net/cache/triple4169_new.json",
    #             template_path=r"concept_net/cache/template4169_new.json",
    #             save_path=r"concept_net/cache/sentence4169_new.json")
