# encoding:utf-8
import sys
import os
import json
import datetime

reload(sys)
sys.setdefaultencoding('utf-8')

project_path = os.path.abspath('.') + '/'

raw_data_prefix = project_path + r"raw_data/"
data_prefix = project_path + r"data/"
features_prefix = project_path + r"data/features/"
result_prefix = project_path + r"result/"
models_prefix = project_path + r"data/models/"
txt_prefix = project_path + r"data/txt/"
jobs2_path = raw_data_prefix + "jobs.keys.txt"
jobs2_path2 = raw_data_prefix + "jobs.keys2.txt"

## 自动创建不存在的文件夹
if os.path.exists(data_prefix) is False:
    os.mkdir(data_prefix)
if os.path.exists(features_prefix) is False:
    os.mkdir(features_prefix)
if os.path.exists(models_prefix) is False:
    os.mkdir(models_prefix)
if os.path.exists(txt_prefix) is False:
    os.mkdir(txt_prefix)
if os.path.exists(raw_data_prefix) is False:
    os.mkdir(raw_data_prefix)

if os.path.exists(raw_data_prefix+'practice.json') is False:
    from colorama import init, Fore

    init(autoreset=True)
    # 通过使用autoreset参数可以让变色效果只对当前输出起作用，输出完成后颜色恢复默认设置
    print Fore.RED+'please download the practice json from'
    print 'http://pan.baidu.com/s/1gfLkfj9'
    print 'practice.json to file named as raw_data'


def getJsonFile(json_fname):
    json_file = file(json_fname, "r")
    json_vector = []
    for line in json_file:
        person_info = json.loads(line)
        json_vector.append(person_info)
    return json_vector


def storeData(distri_dict, key, major, degree_id):
    if major not in distri_dict[key]:
        distri_dict[key].setdefault(major, {})
    if degree_id not in distri_dict[key][major]:
        distri_dict[key][major].setdefault(degree_id, 0)
    distri_dict[key][major][degree_id] += 1
    return distri_dict


def getHashDict(fname):
    jobs_dict = {}
    for line in open(fname, "r"):
        if line.find("技术支持") != -1: line = "技术支持"
        arr = line.strip().split(":")
        if len(line) > 0: jobs_dict.setdefault(arr[0], arr[1])
    return jobs_dict


def getRevHashDict(fname):
    jobs_dict = {}
    for line in open(fname, "r"):
        if line.find("技术支持") != -1: line = "技术支持"
        arr = line.strip().split(":")
        jobs_dict.setdefault(str(arr[0]), str(arr[0]))
        if len(arr) >= 2:
            job_list = arr[1].split(",")
            for i in xrange(0, len(job_list)):
                jobs_dict.setdefault(str(job_list[i]), str(arr[0]))
    return jobs_dict


def score_lists(list_1, list_2):
    count = 0
    total = len(list_1)
    print total
    for i in range(total):
        if list_1[i] == list_2[i]:
            count += 1
    return str(float(count) / total)


def maxDictElement(sour_dict):
    return sorted(sour_dict.items(), lambda a, b: -cmp(a[1], b[1]))[0][0]


# 以字典方式读取文件
def read_dict(file_path):
    file_r = open(file_path, 'r')
    import pickle

    dic_t = dict(pickle.load(file_r))
    file_r.close()
    return dic_t


# 以文件方式读取文件
def read_file(file_path):
    try:
        file_r = open(file_path, 'r')
        import pickle

        dic_t = pickle.load(file_r)
        file_r.close()
        return dic_t
    except Exception, e:
        print e.message
        return {}


# 文件写出，主要是字典文件
def write_dic(obj, file_path):
    try:
        file_w = open(file_path, 'w')
        import pickle

        pickle.dump(obj, file_w)
        file_w.close()
    except Exception, e:
        print e.message


# 读取json 文件
def read_json(inputJsonFile):
    list_json = []
    fin = open(inputJsonFile, 'r')
    for eachLine in fin:
        line = eachLine.strip().decode('utf-8')  # 去除每行首位可能的空格，并且转为Unicode进行处理
        line = line.strip(',')  # 去除Json文件每行大括号后的逗号
        js = None
        try:
            js = json.loads(line.strip())  # 加载Json文件
            list_json.append(js)
        except Exception, e:
            eachLine = eachLine[eachLine.find("{"):]
            line = eachLine.strip().decode('utf-8')
            line = line.strip(',')
            list_json.append(json.loads(line.strip()))
            continue
    fin.close()  # 关闭文件
    return list_json


# 检测json读取是否正常
def show_test(inputJsonFile):
    list_json = []
    fin = open(inputJsonFile, 'r')
    a = 0
    for eachLine in fin:
        line = eachLine.strip().decode('utf-8')  # 去除每行首位可能的空格，并且转为Unicode进行处理
        line = line.strip(',')  # 去除Json文件每行大括号后的逗号
        js = None
        try:
            js = json.loads(line)  # 加载Json文件
            list_json.append(js)
            if a < 10:
                a += 1
                print js
        except Exception, e:
            print 'bad line'
            continue
    fin.close()  # 关闭文件
    return list_json


# 读取 json中的major，主要是验证下属性读取是否顺利
def get_major_dict(list_json):
    m_dict = {}
    for i in list_json:
        if m_dict.has_key(i['major']):
            m_dict[i['major']] += 1
            continue
        else:
            m_dict.setdefault(i['major'], 0)
    return m_dict


def work_time(start_date, end_date):
    start_date = start_date.strip()
    end_date = end_date.strip()
    if u'今' in end_date or 'Present' in end_date or u'其他' in end_date:
        end_date = '2015-05'
    a = str(start_date).split('-')[0]
    b = str(start_date).split('-')[1]
    c = str(end_date).split('-')[0]
    d = str(end_date).split('-')[1]
    d1 = datetime.datetime(int(c), int(d), 1)
    d2 = datetime.datetime(int(a), int(b), 1)
    return round(float((d1 - d2).days) / 30.5 / 12, 2)


position_name_list = \
    [u'技术支持',
     u'开发工程师',
     u'质量(QA/QC)',
     u'软件测试',
     u'机械工程师',
     u'会计',
     u'财务',
     u'项目经理',
     u'客服经理',
     u'客服',
     u'销售总监',
     u'销售经理',
     u'销售专员',
     u'市场总监',
     u'市场经理',
     u'市场专员',
     u'采购总监',
     u'采购经理',
     u'采购助理',
     u'生产总监',
     u'生产经理',
     u'生产专员',
     u'物流总监',
     u'物流经理',
     u'物流专员',
     u'运营总监',
     u'运营经理',
     u'运营专员',
     u'后勤主管',
     u'后勤专员',
     u'人力资源经理',
     u'人力资源专员']

position_name_list2 = \
    ['技术支持',
     '开发工程师',
     '质量(QA/QC)',
     '软件测试',
     '机械工程师',
     '会计',
     '财务',
     '项目经理',
     '客服经理',
     '客服',
     '销售总监',
     '销售经理',
     '销售专员',
     '市场总监',
     '市场经理',
     '市场专员',
     '采购总监',
     '采购经理',
     '采购助理',
     '生产总监',
     '生产经理',
     '生产专员',
     '物流总监',
     '物流经理',
     '物流专员',
     '运营总监',
     '运营经理',
     '运营专员',
     '后勤主管',
     '后勤专员',
     '人力资源经理',
     '人力资源专员']
