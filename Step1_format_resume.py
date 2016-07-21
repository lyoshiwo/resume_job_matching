# coding:utf-8
import util
import json
import sys
import os

reload(sys)
sys.setdefaultencoding('utf-8')


# 每个字段都采用这个函数进行初处理步的
# Every string field need be processed by this function
# 包括 1）缺失值归一为"-" 2）小写字母转为大写字母 3）去除空格
# Process includes 1)Empty <- "_" 2)upper 3)" "<-''
def global_default(s):
    if s is None or s == "null" or s == "" or s == 'None':
        s = "-"

    s = s.upper()

    if s.find(' ') != -1:
        # print s
        s = s.replace(' ', '')

    return s


# 并列词或者并列字符全部改为"/"
# copulative words need be replaced by '/'
def side_by_side(sou_str):
    replace_list = ["以及", "及", "，", ",", "和", "与", "&", "、", "\\"]
    for word in replace_list:
        if sou_str.find(word) != -1:
            sou_str = sou_str.replace(word, '/')
    while sou_str.find("//") != -1:
        sou_str = sou_str.replace('//', '/')
    return sou_str


position_name_dict, department_dict = {}, {}


# drop string in () or in （） of position_name
def get_all(json_data):
    for i in xrange(len(json_data)):
        jobs = json_data[i]["workExperienceList"]
        for job_id in xrange(len(jobs)):
            job = jobs[job_id]
            if job is None:
                continue
            department = str(job["department"])
            for part in department.split('/'):
                if part == ' ' or part == '':
                    continue
                if part.find('(') != -1:
                    part = part[:part.find('(')]
                if part.find('（') != -1:
                    part = part[:part.find('（')]
                department_dict[part] = department_dict.setdefault(part, 0) + 1
            position_name = str(job["position_name"])
            if position_name != "质量(QA/QC)":
                for part in position_name.split('/'):
                    if part == ' ' or part == '':
                        continue
                    if part.find('(') != -1:
                        part = part[:part.find('(')]
                    if part.find('（') != -1:
                        part = part[:part.find('（')]
                    position_name_dict[part] = position_name_dict.setdefault(part, 0) + 1
            else:
                position_name_dict["质量(QA/QC)"] = position_name_dict.setdefault("质量(QA/QC)", 0) + 1


# raw起始的单词 与原始数据操作有关
# if word groups begin with "raw", they are related to original json data.
def raw_process(json_data):
    for i in xrange(len(json_data)):
        # gender字段
        gender = json_data[i]["gender"]
        if gender is not None and gender.lower() == "female":
            gender = "女"
        if gender is not None and gender.lower() == "male":
            gender = "男"
        json_data[i]["gender"] = gender
        # age 字段
        age = global_default(str(json_data[i]["age"]))
        age = age.replace("岁", "")
        if age.isdigit() and int(age) < 16:
            age = "16"
        elif age.isdigit() is False:
            age = "16"
        json_data[i]["age"] = age
        # major字段
        major = global_default(str(json_data[i]["major"]))
        if major.find("MBA") != -1:
            major = "MBA"
        if major.find("会计学") != -1:
            major = "会计"
        if major == "金融":
            major = "金融学"
        if major.find("财务与会计") != -1:
            major = "财务会计"
        if major.find("软件") != -1 and major.find("测试") == -1:
            major = "软件工程"
        major = major.replace("方向", "")

        major = global_default(side_by_side(major))
        json_data[i]["major"] = major
        # 工作经历 workExperience
        jobs = json_data[i]["workExperienceList"]
        for job_id in xrange(len(jobs)):
            job = jobs[job_id]
            if job is not None:
                type = global_default(str(job["type"]))
                industry = global_default(str(job["industry"]))
                department = global_default(str(job["department"]))
                position_name = global_default(str(job["position_name"]))

                if (position_name.find("兼职") != -1 and \
                            (position_name.find('兼') == 0 or position_name.find('职') == len(position_name) - 1) \
                            and (type is None or type == 'null' or type == '-')) \
                        or type == "兼职-":
                    type = "兼职-兼职"
                type = type.replace(' ', '')
                json_data[i]["workExperienceList"][job_id]["type"] = type
                if position_name.find("兼职") != -1 and position_name != '兼职':
                    position_name = position_name.replace('兼职', '/')
                if position_name.find("兼职") == -1 and position_name.find("兼") != -1:
                    position_name = position_name.replace('兼', '/')

                position_name = global_default(side_by_side(position_name))

                if position_name.find(">") != -1:
                    position_name = position_name.replace(">>", '>')
                    position_name = position_name.replace("==>", '-->')
                    position_name = position_name.split("->")[-1]
                while position_name.find("---") != -1:
                    position_name = position_name.replace('---', '--')
                while position_name.find("--") != -1:
                    position_name = position_name.split('--')[-1]
                if position_name.find("实习") != -1:
                    position_name = "实习生"
                if position_name.find('/') != -1 and position_name[0] == '/': position_name = position_name[1:]
                if position_name.find('/') != -1 and position_name[-1] == '/': position_name = position_name[:-1]

                json_data[i]["workExperienceList"][job_id]["position_name"] = position_name
                # 部门进行归一
                department = global_default(side_by_side(department))
                department = department.replace("一部", "")
                department = department.replace("二部", "")
                department = department.replace("三部", "")
                department = department.replace("四部", "")

                department = department.replace("部门", "部")
                department = department.replace("部", "")
                if department == "行政人事":
                    department = "人事行政"
                if department == "总经理室" or department == "总经办":
                    department = "总经理办公室"
                while department.find("--") != -1:
                    department = department.split('--')[0]
                json_data[i]["workExperienceList"][job_id]["department"] = department

                industry = side_by_side(industry)
                json_data[i]["workExperienceList"][job_id]["industry"] = industry

    get_all(json_data)
    return json_data


# 去除括号，如果括号中的内容在括号外出现在limit次以上，保留并作为并列项，否则删除
# the values in () or （）,if they appearance more than limit times, keep them with mark / /, or drop them.
def erase_bracket(sour, compare_dict, limit):
    bracket_list = [["(", ")"], ["（", "）"]]
    for bracket in bracket_list:
        first, end = sour.find(bracket[0]), sour.find(bracket[1])
        if first != -1 and end != -1:
            content = sour[first + len(bracket[0]): end]
            if content in compare_dict and compare_dict[content] > limit:
                sour = sour[: first] + '/' + content + '/' + sour[end + len(bracket[1]):]
            else:
                sour = sour[: first] + sour[end + len(bracket[1]):]
        elif first != -1 or end != -1:
            sour = sour.replace(bracket[0], "")
            sour = sour.replace(bracket[1], "")

    if sour.find('/') != -1 and sour[0] == '/': sour = sour[1:]
    if sour.find('/') != -1 and sour[-1] == '/': sour = sour[:-1]
    return sour


# 对每个字段的括号进行处理
def bracket_process(json_data):
    for i in xrange(len(json_data)):
        jobs = json_data[i]["workExperienceList"]
        for job_id in xrange(len(jobs)):
            job = jobs[job_id]
            if job is None:
                continue
            department = str(job["department"])
            position_name = str(job["position_name"])
            department = erase_bracket(department, department_dict, 5)
            if position_name != "质量(QA/QC)":
                position_name = erase_bracket(position_name, position_name_dict, 10)
            position_name_list = position_name.split('/')
            for part in position_name_list:
                if position_name != "质量(QA/QC)" and part in util.position_name_list and part != position_name:
                    position_name = part
                    break
            json_data[i]["workExperienceList"][job_id]["department"] = department
            json_data[i]["workExperienceList"][job_id]["position_name"] = position_name
    return json_data


# 对position-type和position-department对进行统计
# 用来在type和department未知的情况下进行填充
key_list = ["type", "department"]


def get_missing_value_dict(trn_json_data, tst_json_data):
    missing_value_dict = {}
    data_list = [trn_json_data, tst_json_data]
    for data in data_list:
        for person in data:
            jobs = person["workExperienceList"]
            for job_id in xrange(len(jobs)):
                if job_id == 1: continue

                job = jobs[job_id]
                type = job["type"]
                department = job["department"]
                position_name = job["position_name"]

                if type is not None and type != 'null' and type != '-' and len(type) >= 2:
                    missing_value_dict["type"][position_name][type] = missing_value_dict.setdefault("type",
                                                                                                    {}).setdefault(
                        position_name, {}).setdefault(type, 0) + 1
                if department is not None and department != 'null' and department != '-' and len(department) >= 2:
                    missing_value_dict["department"][position_name][department] = missing_value_dict.setdefault(
                        "department", {}).setdefault(position_name, {}).setdefault(department, 0) + 1

    for key in key_list:
        for position_name in missing_value_dict[key]:
            tmp_list = missing_value_dict[key][position_name]
            tmp_list = sorted(tmp_list, lambda a, b: -cmp(a[1], b[1]))
            missing_value_dict[key][position_name] = tmp_list[0]
    return missing_value_dict


# 利用我们得到的{position_name,department,type}字典对department和type缺失的字段进行填充
def fill_mssing_value(json_data, missing_value_dict):
    for i in xrange(len(json_data)):
        jobs = json_data[i]["workExperienceList"]
        for job_id in xrange(len(jobs)):

            job = jobs[job_id]
            if job is not None:
                type = job["type"]
                department = job["department"]
                position_name = job["position_name"]

                if (type is None or type == 'null' or type == '-' or len(type) < 2) and position_name in \
                        missing_value_dict["type"]:
                    type = missing_value_dict["type"][position_name]
                if (department is None or department == 'null' or department == '-' or len(
                        department) < 2) and position_name in missing_value_dict["department"]:
                    department = missing_value_dict["department"][position_name]

                json_data[i]["workExperienceList"][job_id]["type"] = type
                json_data[i]["workExperienceList"][job_id]["department"] = department

    return json_data


def write_json_line(line, f_out):
    line = line.strip().decode('utf-8')
    if line[0] != "{":
        line = line[line.find("{"):]
    try:
        json_line = json.loads(line)  # 加载Json文件
    except Exception, e:
        print 'bad line'
        return
    outStr = json.dumps(json_line, ensure_ascii=False)  # 处理完之后重新转为Json格式
    f_out.write(outStr.encode('utf-8') + '\n')  # 写回到一个新的Json文件中去


def write_json_dict(person, f_out):
    outStr = json.dumps(person, ensure_ascii=False)  # 处理完之后重新转为Json格式
    f_out.write(outStr.encode('utf-8') + '\n')  # 写回到一个新的Json文件中去


if __name__ == "__main__":
    print 'may take several minutes, please wait'

    trn_path = util.raw_data_prefix + "practice.json"
    if os.path.exists(trn_path) is False:
        exit(-1)
    trn_json_data = util.getJsonFile(trn_path)

    new_trn_json_data = raw_process(trn_json_data)

    new_trn_json_data = bracket_process(new_trn_json_data)

    trn_file = open(util.raw_data_prefix + "resume.json", "w")
    for person in new_trn_json_data:
        write_json_dict(person, trn_file)
    trn_file.close()
    print 'resume format done'
