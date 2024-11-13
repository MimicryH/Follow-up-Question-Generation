from FollowUps.concept_net.ref import *
import pandas as pd


def check_validation(input_text, triple):
    if triple[5] == '' or triple[5] != '0':
        return True
    else:
        return False
        # test_type = int(triple[5])

    # if test_type == 0:
    #     return False
    # elif test_type == 1:
    #     if triple[0] in input_text:
    #         return True
    #     else:
    #         return False
    # elif test_type == 2:
    #     if triple[3] in input_text:
    #         return True
    #     else:
    #         return False
    # elif test_type == 3:
    #     if triple[0] in input_text and triple[3] in input_text:
    #         return True
    #     else:
    #         return False


def getTempPOS(template, pattern):
    temp_pos = None
    obj = re.compile(pattern).search(template)
    if obj:
        temp_pos = obj.group(1)
    return temp_pos


class CN_generator(object):
    def __init__(self, triple_path=r"FollowUps/concept_net/cache/triple4169_new.json",
                 sentence_path=r"FollowUps/concept_net/cache/sentence4169_new.json"):
        self.get_fluency = get_fluency
        with open(sentence_path, encoding='utf-8') as f:
            self.sentence_ref = json.load(f)
        with open(triple_path, encoding='utf-8') as f:
            self.triples_ref = json.load(f)

    def match_entities(self, text):
        entities = []
        for keyword in self.triples_ref.keys():
            if keyword != '':
                if keyword in text and keyword not in entities:
                    entities.append(keyword)
        return entities

    def get_triples_entities(self, entities):
        triple_entity_pairs = []
        for entity in entities:
            for triple in self.triples_ref[entity]:
                if triple not in triple_entity_pairs:
                    triple_entity_pairs.append([triple, entity])
        res_triples = []
        res_t_e = []
        # remove the repeated triples
        for t_e in triple_entity_pairs:
            if t_e[0] not in res_triples:
                res_t_e.append(t_e)
                res_triples.append(t_e[0])
        return res_t_e

    def genSentences(self, input_text, triple_entity_list):
        res = pd.DataFrame(columns=['template', 'triple', 'entity','question', 'flue'])
        print('genSentences', input_text)
        for t_e in triple_entity_list:
            triple_key = "-".join(t_e[0])
            # print(self.sentence_ref[triple_key])
            if triple_key not in self.sentence_ref.keys():
                print(triple_key + 'does not exist.....')
            else:
                for info in self.sentence_ref[triple_key]:
                    template, triple, flue, sent = info
                    if check_validation(input_text, triple):
                        if '<W0>' in sent:
                            if triple[0] in input_text:
                                sent = re.sub(r'<W0(.*?)>', triple[3], sent)
                            else:
                                sent = re.sub(r'<W0(.*?)>', triple[0], sent)
                        if sent not in res['question']:
                            res = res.append({"template": template, "triple": triple, "entity": t_e[1],"question": sent,
                                              'flue': float(flue)}, ignore_index=True)
                    else:
                        pass
                    # print(triple, "cannot match", input_text)
        return res
