# encoding:utf-8
import json
import sys

import util

reload(sys)
sys.setdefaultencoding('utf-8')

resumes = util.read_json(util.raw_data_prefix + 'resume.json')
list_count = [0, 0, 0, 0, 0, 0, 0]
resumes_clean = []
# major degree age id workExperienceList salary end_date industry department position_name start_date size
for i in resumes:
    z = len(i['workExperienceList'])
    for k in range(z):
        if i['workExperienceList'][k]['position_name'] in util.position_name_list:
            if z - k - 1 > 1:
                i['workExperienceList'] = i['workExperienceList'][k:]
                resumes_clean.append(i)
                break
# 47346 [10809, 11845, 30820, 11376, 3586, 1061, 503]
print len(resumes_clean)

abc = set()
kk = resumes_clean
for i in kk:
    work_list = i['workExperienceList']
    for z in work_list:
        abc.add(z['position_name'])
print len(abc)


def write_json_dict(person, f_out):
    # 处理完之后重新转为Json格式
    f_out.write(json.dumps(person, ensure_ascii=False).encode('utf-8') + '\n')  # 写回到一个新的Json文件中去


trn_file = open(util.data_prefix + "resume_clean.json", "w")
for person in resumes_clean:
    write_json_dict(person, trn_file)
trn_file.close()

