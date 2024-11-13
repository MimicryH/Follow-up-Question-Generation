import difflib
# from interviewDS.Followup.sim_hownet import *
# hownet = SimHownet()
def check_repeat(s1,s2):
    score= difflib.SequenceMatcher(None, s1, s2).quick_ratio()
    print (score)
    if score>0.38:
        return False
    else:
        return True


s1="那你认识就是你高中高考以前有有朋友或者同学早恋吗？"
s2="学校里有同学早恋吗？"
check_repeat(s1,s2)