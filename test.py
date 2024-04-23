'''
Author: huang
Date: 2024-04-23 17:06:30
LastEditors: Do not edit
LastEditTime: 2024-04-23 17:06:36
FilePath: \Image_Recognition_WebGUI-main\test.py
'''
from pywebio.input import input, TEXT, FLOAT, input_update, input_control, textarea
from pywebio.output import put_text, put_tabs, put_table, put_file, put_code
from pywebio import start_server


def cal_jaccard():
    sen_1 = input("请输入第一个句子：", type=TEXT)
    sen_2 = input("请输入第二个句子：", type=TEXT)
    try:
        sent_intersection = list(set(list(sen_1)).intersection(set(list(sen_2))))
        sent_union = list(set(list(sen_1)).union(set(list(sen_2))))
        score_jaccard = round(float(len(sent_intersection) / len(sent_union)), 6)
    except:
        score_jaccard = 0.0
    res = {"result": score_jaccard}
    data = {"code": 200, "data": res, "message": "success"}
    # put_text(data)
    put_text(data)


if __name__ == '__main__':
    start_server(cal_jaccard, port=8832)
