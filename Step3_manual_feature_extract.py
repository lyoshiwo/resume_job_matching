# encoding=utf-8
import pandas as pd
import jieba
import jieba.analyse
import codecs
import json
import sys
import util

reload(sys)
sys.setdefaultencoding('utf-8')

columns = \
    [
        "id",
        "major", "age", "gender",
        "isenglish", "isjunior", "isbachelor", "ismaster", "isintern",
        "total_previous_job",
        "last_salary", "last_size", "last_position_name", "last_industry", "last_type", "last_type1", "last_department",
        "last_start_year", "last_start_month", "last_end_year", "last_end_month", "last_interval_month",
        "third_salary", "third_size", "third_position_name", "third_industry", "third_type", "third_type1",
        "third_department",
        "third_start_year", "third_start_month", "third_end_year", "third_end_month", "third_interval_month",
        "first_salary", "first_size", "first_position_name", "first_industry", "first_type", "first_type1",
        "first_department",
        "first_start_year", "first_start_month", "first_end_year", "first_end_month", "first_interval_month",
        "last3_interval_month", "diff_last3_size", "diff_last3_salary", "diff_last3_industry",
        "diff_last3_position_name",
        "total_interval_month", "diff_salary", "diff_size", "diff_industry", "diff_position_name", "start_working_age",
        "pre_largest_size", "pre_largest_salary",
        "major_1",
        "last_position_name_1", "last_department_1",
        "third_position_name_1", "third_department_1",
        "first_position_name_1", "first_department_1",
        "major_2",
        "last_position_name_2", "last_department_2",
        "third_position_name_2", "third_department_2",
        "first_position_name_2", "first_department_2",
        "rev_working_age", "pre_working_month", "pre_interval_month",
        "total_position_name", "total_size", "total_salry", "total_type", "total_department", "total_industry",


        "pre_least_size",
        "pre_least_salary",
        "pre_size1",
        "pre_size2",
        "pre_size3",
        "pre_size4",
        "pre_size5",
        "pre_size6",
        "pre_size7",
        "pre_salary1",
        "pre_salary2",
        "pre_salary3",
        "pre_salary4",
        "pre_salary5",
        "pre_salary6",
        "pre_salary7",

        "promotion_size",
        "promotion_salary",
        "decrease_size",
        "decrease_salar",

        "predict_degree", "predict_salary", "predict_size", "predict_position_name"
    ]
print len(columns)


def parse_date(date_str):
    t = date_str.split('-')
    year, month = int(t[0]), int(t[1])
    return year, month


def parse_start_end_date(date_str_start, date_str_end):
    if date_str_end == '至今' or date_str_end == '今' or date_str_end == None or date_str_end == 'Present':
        end_year, end_month = 2015, 9
    else:
        if date_str_end == u'至今':
            print date_str_end
        end_year, end_month = parse_date(date_str_end)

    if date_str_start == None:
        if end_month > 6:
            start_year, start_month = end_year, end_month - 6
        else:
            start_year, start_month = end_year - 1, 12 - (6 - end_month)
    else:
        start_year, start_month = parse_date(date_str_start)

    return start_year, start_month, end_year, end_month


def extract_major(sentens):
    if sentens == None or sentens == "-": return "", ""

    bracket_list = [["(", ")"], ["（", "）"]]
    for bracket in bracket_list:
        l, r = -1, -1
        if sentens.find(bracket[0]) != -1:
            l = sentens.find(bracket[0])
        if sentens.find(bracket[1]) != -1:
            r = sentens.find(bracket[1])

        if l != -1 and r != -1:
            return sentens[:l], sentens[l + len(bracket[0]):r]
        else:
            return sentens, sentens


def extract_title(sentens):
    if sentens == None or sentens == "-": return "", ""

    words = list(jieba.cut(sentens, cut_all=False))
    word1, word2 = "", ""
    if len(words) > 0:  word1 = words[0]
    if len(words) > 0:  word2 = words[-1]

    return word1, word2


def getFeature_Part1(f_in_name):
    positions = []
    f_in = codecs.open(f_in_name, "r", "utf-8")
    for line in f_in:
        # basic features
        person = json.loads(line)
        id = person['id']
        major = person['major']
        age = int(person['age'])
        gender = person['gender']

        ## major's key word
        isenglish = 0
        isjunior = 0
        isbachelor = 0
        ismaster = 0
        if major.isalpha() == True: isenglish = 1
        for key in ["大专", "中专", "高中", "初中", "中技", "职高", "中职", "高职"]:
            if line.find(key) != -1:  isjunior = 1
        for key in ["本科"]:
            if line.find(key) != -1: isbachelor = 1
        for key in ["硕士", "博士", "MBA", "EMBA"]:
            if line.find(key) != -1: isbachelor = 1

        work_exps_list = person['workExperienceList']
        total_previous_job = len(work_exps_list) - 1

        ## extract key words of person
        total_pos_dict = {}
        total_size_dict = {}
        total_salary_dict = {}
        total_type_dict = {}
        total_industry_dict = {}
        total_department_dict = {}
        test_str = str(major) + ","
        for i in xrange(len(work_exps_list)):
            if i == 1: continue  # rev second job
            if work_exps_list[i]["position_name"].isalpha() == True: isenglish = 1
            test_str += "%s,%s,%s,%s" % (work_exps_list[i]["industry"], \
                                         work_exps_list[i]["department"], work_exps_list[i]["position_name"],
                                         work_exps_list[i]["type"])
            total_size_dict.setdefault(work_exps_list[i]["size"], 0)
            total_salary_dict.setdefault(work_exps_list[i]["salary"], 0)
            total_type_dict.setdefault(work_exps_list[i]["type"], 0)
            total_industry_dict.setdefault(work_exps_list[i]["industry"], 0)
            total_department_dict.setdefault(work_exps_list[i]["department"], 0)
            total_pos_dict.setdefault(work_exps_list[i]["position_name"], 0)
        isintern = 0 if test_str.find("实习") == -1 else 1

        ## total different jobs
        total_position_name = len(total_pos_dict)
        total_size = len(total_size_dict)
        total_salry = len(total_salary_dict)
        total_type = len(total_type_dict)
        total_department = len(total_department_dict)
        total_industry = len(total_industry_dict)
        # extended features
        # last job features
        last_job = work_exps_list[1]
        last_salary = int(last_job['salary'])
        last_size = int(last_job['size'])
        last_position_name = last_job['position_name']
        last_industry = last_job['industry']
        last_type = last_job['type']
        last_type1 = last_type.split("-")[0]
        last_department = last_job['department']
        last_end_date = last_job['end_date']
        last_start_date = last_job['start_date']
        last_start_year, last_start_month, last_end_year, last_end_month = parse_start_end_date(last_start_date,
                                                                                                last_end_date)
        last_interval_month = (last_end_year - last_start_year) * 12 + (last_end_month - last_start_month)

        # last job features
        third_job = work_exps_list[2]
        third_salary = int(third_job['salary'])
        third_size = int(third_job['size'])
        third_position_name = third_job['position_name']
        third_industry = third_job['industry']
        third_type = third_job['type']
        third_type1 = third_type.split("-")[0]
        third_department = third_job['department']
        third_end_date = third_job['end_date']
        third_start_date = third_job['start_date']
        third_start_year, third_start_month, third_end_year, third_end_month = parse_start_end_date(third_start_date,
                                                                                                    third_end_date)
        third_interval_month = (third_end_year - third_start_year) * 12 + (third_end_month - third_start_month)

        # first job features
        first_job = work_exps_list[-1]
        first_salary = int(first_job['salary'])
        first_size = int(first_job['size'])
        first_position_name = first_job['position_name']
        first_industry = first_job['industry']
        first_type = first_job['type']
        first_type1 = first_type.split("-")[0]
        first_department = first_job['department']
        first_end_date = first_job['end_date']
        first_start_date = first_job['start_date']
        first_start_year, first_start_month, first_end_year, first_end_month = parse_start_end_date(first_start_date,
                                                                                                    first_end_date)
        first_interval_month = (first_end_year - first_start_year) * 12 + (first_end_month - first_start_month)

        # all workexperience feature

        pre_largest_salary = -1
        pre_largest_size = -1
        pre_least_salary = -1
        pre_least_size = -1
        pre_working_month = 0
        pre_size_list = [0, 0, 0, 0, 0, 0, 0, 0]
        pre_salary_list = [0, 0, 0, 0, 0, 0, 0]
        promotion_size = 0
        promotion_salary = 0
        decrease_size = 0
        decrease_salary = 0
        for work in work_exps_list[1:]:
            if work != None:
                if pre_largest_size == -1 or pre_largest_size < work["size"]:  pre_largest_size = work["size"]
                if pre_largest_salary == -1 or pre_largest_salary < work["salary"]: pre_largest_salary = work["salary"]
                if pre_least_size == -1 or pre_least_size > work["size"]:  pre_least_size = work["size"]
                if pre_least_salary == -1 or pre_least_salary > work["salary"]: pre_least_salary = work["salary"]
                start_year, start_month, end_year, end_month = parse_start_end_date(work['start_date'],
                                                                                    work['end_date'])
                pre_working_month += (end_year - start_year) * 12 + (end_month - start_month)
                if work["size"] != None:
                    pre_size_list[work["size"]] += 1
                if work["salary"] != None:
                    pre_salary_list[work["salary"]] += 1
                if i < len(work_exps_list) - 1:
                    if work_exps_list[i + 1]["size"] != None and work_exps_list[i]["size"] > work_exps_list[i + 1][
                        "size"]:
                        promotion_size += 1
                    if work_exps_list[i + 1]["size"] != None and work_exps_list[i]["size"] < work_exps_list[i + 1][
                        "size"]:
                        decrease_size += 1
                    if work_exps_list[i + 1]["salary"] != None and work_exps_list[i]["salary"] > work_exps_list[i + 1][
                        "salary"]:
                        promotion_salary += 1
                    if work_exps_list[i + 1]["salary"] != None and work_exps_list[i]["salary"] < work_exps_list[i + 1][
                        "salary"]:
                        decrease_salary += 1

        pre_size1 = pre_size_list[1]
        pre_size2 = pre_size_list[2]
        pre_size3 = pre_size_list[3]
        pre_size4 = pre_size_list[4]
        pre_size5 = pre_size_list[5]
        pre_size6 = pre_size_list[6]
        pre_size7 = pre_size_list[7]
        pre_salary1 = pre_salary_list[0]
        pre_salary2 = pre_salary_list[1]
        pre_salary3 = pre_salary_list[2]
        pre_salary4 = pre_salary_list[3]
        pre_salary5 = pre_salary_list[4]
        pre_salary6 = pre_salary_list[5]
        pre_salary7 = pre_salary_list[6]
        # last and third diff
        last3_interval_month = (last_end_year - third_start_year) * 12 + (last_end_month - third_start_month)
        diff_last3_size = last_size - third_size
        diff_last3_salary = last_salary - third_salary
        diff_last3_industry = last_industry == third_industry
        diff_last3_position_name = last_position_name == third_position_name

        # total difference
        total_interval_month = (last_end_year - first_start_year) * 12 + (last_end_month - first_start_month)
        diff_size = last_size - first_size
        diff_salary = last_salary - first_salary
        diff_industry = last_industry == first_industry
        diff_position_name = last_position_name == first_position_name

        start_working_age = (age * 12 - total_interval_month) / 12
        rev_working_age = start_working_age + pre_working_month / 12
        if age < 0:
            print age, total_interval_month, start_working_age
        pre_interval_month = (third_end_year - first_start_year) * 12 + (third_end_month - first_start_month)

        major_1, major_2 = extract_major(major)
        last_position_name_1, last_position_name_2 = extract_title(last_position_name)
        last_department_1, last_department_2 = extract_title(last_department)
        third_position_name_1, third_position_name_2 = extract_title(third_position_name)
        third_department_1, third_department_2 = extract_title(third_department)
        first_position_name_1, first_position_name_2 = extract_title(first_position_name)
        first_department_1, first_department_2 = extract_title(first_department)

        # prefict features
        # train
        if work_exps_list[0] != None:
            predict_job = work_exps_list[0]
            predict_degree = int(person['degree'])
            predict_salary = int(predict_job['salary'])
            predict_size = int(predict_job['size'])
            predict_position_name = predict_job['position_name']
        else:
            predict_degree = 0
            predict_salary = 0
            predict_size = 0
            predict_position_name = "销售经理"

        resume = [
            id,
            major,
            age,
            gender,

            isenglish,
            isjunior,
            isbachelor,
            ismaster,
            isintern,

            total_previous_job,
            last_salary,
            last_size,
            last_position_name,
            last_industry,
            last_type,
            last_type1,
            last_department,
            last_start_year,
            last_start_month,
            last_end_year,
            last_end_month,
            last_interval_month,
            third_salary,
            third_size,
            third_position_name,
            third_industry,
            third_type,
            third_type1,
            third_department,
            third_start_year,
            third_start_month,
            third_end_year,
            third_end_month,
            third_interval_month,
            first_salary,
            first_size,
            first_position_name,
            first_industry,
            first_type,
            first_type1,
            first_department,
            first_start_year,
            first_start_month,
            first_end_year,
            first_end_month,
            first_interval_month,
            last3_interval_month,
            diff_last3_size,
            diff_last3_salary,
            diff_last3_industry,
            diff_last3_position_name,
            total_interval_month,
            diff_salary,
            diff_size,
            diff_industry,
            diff_position_name,
            start_working_age,
            pre_largest_size,
            pre_largest_salary,

            major_1,
            last_position_name_1,
            last_department_1,
            third_position_name_1,
            third_department_1,
            first_position_name_1,
            first_department_1,
            major_2,
            last_position_name_2,
            last_department_2,
            third_position_name_2,
            third_department_2,
            first_position_name_2,
            first_department_2,

            rev_working_age,
            pre_working_month,
            pre_interval_month,
            total_position_name,
            total_size,
            total_salry,
            total_type,
            total_department,
            total_industry,
            pre_least_size,
            pre_least_salary,
            pre_size1,
            pre_size2,
            pre_size3,
            pre_size4,
            pre_size5,
            pre_size6,
            pre_size7,
            pre_salary1,
            pre_salary2,
            pre_salary3,
            pre_salary4,
            pre_salary5,
            pre_salary6,
            pre_salary7,
            promotion_size,
            promotion_salary,
            decrease_size,
            decrease_salary,

            predict_degree,
            predict_salary,
            predict_size,
            predict_position_name
        ]
        positions.append(resume)
    # form a DataFrame with all resumes
    data_frame = pd.DataFrame(positions)
    data_frame.columns = columns
    return data_frame

print 'start'
data = getFeature_Part1(util.data_prefix + "resume_clean.json")
pd.to_pickle(data, util.features_prefix + 'manual_feature.pkl')
print 'end'
