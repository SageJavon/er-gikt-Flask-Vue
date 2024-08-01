"""
模型预测
"""
import heapq
import torch
import numpy as np
import traceback
from flask import Blueprint, request
from icecream import ic
from scipy import sparse
import json
from decimal import Decimal

import re
import os
import sys
from scipy import sparse
import heapq
# from params import *
import time
import pymysql

from gikt import GIKT
from utils import gen_gikt_graph, build_adj_list
from dataset import UserDataset
from torch.utils.data import DataLoader

# 获取当前脚本的绝对路径
script_dir = os.path.dirname(__file__)

# 使用绝对路径加载模型和数据文件
model_path = os.path.join(script_dir, 'model-100/results.pth')
qq_table_path = os.path.join(script_dir, 'data/qq_table.npz')
qs_table_path = os.path.join(script_dir, 'data/qs_table.npz')
ss_table_path = os.path.join(script_dir, 'data/ss_table.npz')
question2idx_path = os.path.join(script_dir, 'data/question2idx.npy')
idx2question_path = os.path.join(script_dir, 'data/idx2question.npy')

qq_table = sparse.load_npz(qq_table_path).toarray()
# qs_table = sparse.load_npz(qs_table_path).toarray()
qs_table = torch.tensor(sparse.load_npz(qs_table_path).toarray(), dtype=torch.int64, device='cpu')
ss_table = sparse.load_npz(ss_table_path).toarray()
question2idx = np.load(question2idx_path, allow_pickle=True).item()
idx2question = np.load(idx2question_path, allow_pickle=True).item()
num_question = torch.tensor(qs_table.shape[0], device='cpu')
num_skill = torch.tensor(qs_table.shape[1], device='cpu')

q_neighbors_list, s_neighbors_list = build_adj_list()
q_neighbors, s_neighbors = gen_gikt_graph(
    q_neighbors_list, s_neighbors_list, 4, 10)
q_neighbors = torch.tensor(q_neighbors, dtype=torch.int64, device='cpu')
s_neighbors = torch.tensor(s_neighbors, dtype=torch.int64, device='cpu')

# 初始化模型
model = GIKT(
    num_question, num_skill, q_neighbors, s_neighbors, qs_table
).to('cpu')

model.load_state_dict(torch.load(model_path))


def convert_to_dict(cursor, row):
    """Convert MySQL row to dictionary with handling for Decimal type."""
    d = dict(zip(cursor.column_names, row))
    for key, value in d.items():
        if isinstance(value, Decimal):
            d[key] = float(value)  # Convert Decimal to float
    return d


def fetch_exercise_records(student_id):
    conn = None
    cursor = None
    try:
        # 连接到 MySQL 数据库
        conn = pymysql.connect(
            host="mysql.mysql",
            user="root",
            password='pYRGObpCdG',
            database="sage_javon",
            port=3306
        )

        # 创建游标对象，用于执行查询
        cursor = conn.cursor(cursor=pymysql.cursors.DictCursor)  # 使用 dictionary=True 返回字典形式的结果

        # 执行查询 exercise_record 表
        query = "SELECT * FROM exercise_record WHERE student_id = %s"
        cursor.execute(query, (student_id,))
        exercise_records = cursor.fetchall()

        # 准备存放最终结果的列表
        exercise_record_list = []

        # 遍历每条答题记录
        for exercise_record in exercise_records:
            # 假设 record_id 是 exercise_record 表中的记录ID字段
            record_id = exercise_record['id']
            exercise_id = exercise_record['exercise_id']
            score = exercise_record['score']
            submit_time = exercise_record['submit_time']
            exercise_type = exercise_record['type']

            # 查询 exercise_knowledge 表，找到所有相关的 knowledge_id
            query = "SELECT knowledge_id FROM exercise_knowledge WHERE exercise_id = %s"
            cursor.execute(query, (exercise_id,))
            exercise_knowledge_rows = cursor.fetchall()

            # 准备存放知识概念的列表
            knowledge_concepts = []

            # 遍历每个 knowledge_id，并查询 knowledge 表获取对应的 knowledge 字段
            for exercise_knowledge_row in exercise_knowledge_rows:
                knowledge_id = exercise_knowledge_row['knowledge_id']

                # 查询 knowledge 表，获取 knowledge 字段内容
                query = "SELECT knowledge FROM knowledge WHERE id = %s"
                cursor.execute(query, (knowledge_id,))
                knowledge_row = cursor.fetchone()

                if knowledge_row:
                    knowledge = knowledge_row['knowledge']

                    # 构建知识概念对象
                    knowledge_concept = {
                        'knowledgeId': knowledge_id,
                        'knowledge': knowledge
                    }
                    knowledge_concepts.append(knowledge_concept)

            # 转换 submit_time 为 ISO 格式的字符串
            submit_time_str = submit_time.isoformat()

            # 构建每个 exercise record 的 JSON 对象
            exercise_record_json = {
                'recordId': record_id,
                'exerciseId': exercise_id,
                'knowledgeConcept': knowledge_concepts,
                'score': float(score),
                'submitTime': submit_time_str,
                'type': exercise_type
            }

            exercise_record_list.append(exercise_record_json)

        # 构建最终的 JSON 数据结构
        result = {
            'exerciseRecordList': exercise_record_list
        }
        return result

        # 打印 JSON 数据（或者可以返回给调用者）

    except Exception as e:
        print("Error connecting to MySQL database:", e)

    finally:
        # 关闭游标和数据库连接
        if cursor is not None:
            cursor.close()
        if conn is not None:
            conn.close()
        print('MySQL connection closed')

def stu_state_upload(user_state):
    conn=None
    cursor=None
    try:
        # 创建数据库连接
        conn = pymysql.connect(
            host='mysql.mysql',  # 连接主机, 默认127.0.0.1
            user='root',  # 用户名
            passwd='pYRGObpCdG',  # 密码
            port=3306,  # 端口，默认为3306
            database='sage_javon'
        )
        # 生成游标对象 cursor
        cursor = conn.cursor()
        # 查询数据库版本
        cursor.execute("select version()")  # 返回值是查询到的数据数量
        # 通过 fetchall方法获得数据
        data = cursor.fetchone()
        print("Database Version:%s" % data)
        # user_state = np.load('alg/chart_data_100/user_state.npy',allow_pickle=True).item()
        try:
            for k, v in user_state.items():
                sql = "update student set knowledge_state='{}' where id='{}'".format(v, k)
                cursor.execute(sql)
                conn.commit()
        except Exception as err:
            print("更新失败",err)

    except Exception as e:
        print('异常：',e)
    finally:
        if 'cursor' in locals() and cursor:
            cursor.close()
        if conn:
            conn.close()
            print('MySQL connection closed')



def status_update(user_id):
    # user_id
    # user_id = request.args.get("userId")
    print(user_id)
    history_answers = fetch_exercise_records(
        user_id)['exerciseRecordList']  # 获取历史答题记录
    q_list = [question2idx[a['exerciseId']]
              for a in history_answers]  # 获取答题记录的index
    a_list = [1 if a['score'] >
                   0 else 0 for a in history_answers]  # 获取答题answer记录
    max_length = 200
    question_tensor_list = []
    answer_tensor_list = []
    mask_tensor_list = []
    loop_list = [1]

    for q_related in loop_list:
        print(q_related)
        combined_q_list = q_list
        combined_a_list = a_list
        combined_m_list = [1 for i in range(len(combined_q_list))]
        if len(combined_q_list) < max_length:  # 如果序列长度小于最大长度
            combined_q_list += [0] * (max_length - len(combined_q_list))
            combined_a_list += [0] * (max_length - len(combined_a_list))
            combined_m_list += [0] * (max_length - len(combined_m_list))
        elif len(combined_q_list) > max_length:
            combined_q_list = combined_q_list[-max_length:]
            combined_a_list = combined_a_list[-max_length:]
            combined_m_list = combined_m_list[-max_length:]
        question_tensor_list.append(combined_q_list)
        answer_tensor_list.append(combined_a_list)
        mask_tensor_list.append(combined_m_list)

    question_tensor = torch.tensor(question_tensor_list, dtype=torch.int64)
    answer_tensor = torch.tensor(answer_tensor_list, dtype=torch.int64)
    mask_tensor = torch.tensor(mask_tensor_list, dtype=torch.int64)

    # 模型预测
    c_list,state = model(
        question=question_tensor,
        response=answer_tensor,
        mask=mask_tensor,
        DEVICE="cpu"
    )
    print(c_list)
    print(state.shape)
    user_state = {}
    state = state.detach().numpy()
    user_state[user_id]=state

    stu_state_upload(user_state)
    print("更新成功！")

    # return {
    #     'code': 1,
    #     'msg': "knowledge state update success.",
    # }


def predict():
    question_tensor = torch.tensor(np.load('data/user_seq.npy'), dtype=torch.int64)
    # [num_user, max_seq_len] 输入数据
    answer_tensor = torch.tensor(np.load('data/user_res.npy'), dtype=torch.int64)
    # [num_user, max_seq_len] 输入标签
    mask_tensor = torch.tensor(np.load('data/user_mask.npy'), dtype=torch.bool)
    # [num_user, max_seq_len] 有值效记录
    idx2user = np.load('data/idx2user.npy', allow_pickle=True).item()
    keys_array = np.array(list(idx2user.values()))
    array_2d = np.repeat(keys_array[:, np.newaxis], 200, axis=1)
    user_id = torch.tensor(array_2d)
    # 模型预测
    c_list, state = model(
        question=question_tensor,
        response=answer_tensor,
        mask=mask_tensor,
        DEVICE="cpu"
    )
    print(state.shape)
    user_state = {}
    user_id = user_id.detach().numpy()
    knowledge_state = state.detach().numpy()
    for index, row in enumerate(user_id):
        # print(row[0])
        user_state[row[0]] = knowledge_state[index]
    np.save('chart_data_100/user_state.npy', user_state)

# if __name__ == "__main__":
#     predict()