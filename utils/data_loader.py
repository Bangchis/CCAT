import os
import json
import numpy as np
import pandas as pd
from dataset import AdapTestDataset
from setting import params
import numpy as np
import scipy.optimize


def load_data():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data", params.data_name)
    metadata_path = os.path.join(data_dir, "metadata.json")
    concept_map_path = os.path.join(data_dir, "concept_map.json")
    train_path = os.path.join(data_dir, "train_triples.csv")
    test_path = os.path.join(data_dir, "test_triples.csv")

    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    with open(concept_map_path, 'r') as f:
        concept_map = json.load(f)
    train_triplets = pd.read_csv(
        train_path, encoding='utf-8').to_records(index=False)
    test_triplets = pd.read_csv(
        test_path, encoding='utf-8').to_records(index=False)

    train_data = AdapTestDataset(
        train_triplets, metadata['num_train_students'], metadata['num_questions'])
    test_data = AdapTestDataset(
        test_triplets, metadata['num_test_students'], metadata['num_questions'])
    return train_data, test_data, concept_map, metadata


def load_irt_params():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data", params.data_name)
    gamma = np.load(os.path.join(data_dir, "alpha.npy"))
    beta = np.load(os.path.join(data_dir, "beta.npy"))
    return gamma, beta


def load_preprocessed_question_metadata():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data", params.data_name)
    file_path = os.path.join(data_dir, "preprocessed_questions.csv")
    df = pd.read_csv(file_path, encoding='utf-8')

    qmeta = {}
    answer_mapping = {'1': 'A', '2': 'B', '3': 'C', '4': 'D'}

    for _, row in df.iterrows():
        try:
            new_qid = int(row['NewQuestionId'])
        except KeyError:
            new_qid = int(row['new_question_id'])
        try:
            old_qid = int(row['OldQuestionId'])
        except KeyError:
            old_qid = new_qid

        raw_ans = str(row["CorrectAnswer"]).strip()
        correct_answer = answer_mapping.get(raw_ans, raw_ans)
        subjects_raw = str(row.get("SubjectsNames", "")).strip()

        if subjects_raw:
            subjects = [s.strip() for s in subjects_raw.split(',')]
        else:
            subjects = []

        qmeta[new_qid] = {
            "old_question_id": old_qid,
            "correct_answer": correct_answer,
            "options": {},
            "subjects": subjects
        }

    valid_q_ids = sorted(qmeta.keys())
    return qmeta, valid_q_ids


def select_anchor_students(train_data):
    """
    Lấy danh sách các học sinh anchor từ train_data.
    Vì dữ liệu train đã được lọc trong prepare_data.py, nên toàn bộ học sinh trong train_data được coi là anchor.

    Args:
        train_data: Đối tượng AdapTestDataset chứa dữ liệu huấn luyện.

    Returns:
        anchor_ids: List[int] - danh sách các student_id (đã được renumber) của học sinh anchor.
    """
    # train_data.data là dictionary với key là student_id, giá trị là dictionary các câu hỏi đã trả lời.
    anchor_ids = list(train_data.data.keys())
    return anchor_ids


def likelihood(theta, y, a, b):
    """
    Hàm likelihood dùng để tìm nghiệm của theta.
    Args:
        theta: float, khả năng của học sinh.
        y: numpy.array, phản hồi của học sinh (0/1).
        a: numpy.array, tham số phân biệt của các câu hỏi.
        b: numpy.array, tham số độ khó của các câu hỏi.
    Returns:
        Tổng giá trị likelihood.
    """
    return (a * (y - 1/(1 + np.exp(-a*(theta - b))))).sum()


def compute_anchor_thetas(train_data, gamma, beta):
    """
    Ước lượng theta cho mỗi học sinh trong anchor group dựa trên dữ liệu train.

    Args:
        train_data: Đối tượng AdapTestDataset chứa dữ liệu huấn luyện.
        gamma: mảng hoặc list, tham số phân biệt của các câu hỏi (đã được tải từ alpha.npy).
        beta: mảng hoặc list, tham số độ khó của các câu hỏi (đã được tải từ beta.npy).

    Returns:
        anchor_thetas: dict mapping student_id -> theta (float)
    """
    anchor_thetas = {}
    for stu_id, responses in train_data.data.items():
        a_list = []
        b_list = []
        y_list = []
        for qid, correct in responses.items():
            a_list.append(gamma[qid])
            b_list.append(beta[qid])
            y_list.append(correct)
        a_arr = np.array(a_list)
        b_arr = np.array(b_list)
        y_arr = np.array(y_list)
        sol = scipy.optimize.root(likelihood, 0, args=(y_arr, a_arr, b_arr))
        theta = sol.x[0]
        # Clip theta trong khoảng [-4, 4]
        theta = np.clip(theta, -4, 4)
        anchor_thetas[stu_id] = theta
    return anchor_thetas
