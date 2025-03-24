from selection_strategy import MCMC_Selection
from utils.data_loader import load_data, load_irt_params, load_preprocessed_question_metadata
import numpy as np
from flask import session
from setting import params
import os
from collections import defaultdict
import random


def select_next_question(theta, gamma, beta, unanswered, question_meta,
                         difficulty_weight=1,
                         difficulty_sigma=0.9,
                         top_k=8):
    """
    Chọn ngẫu nhiên một trong top‑k câu hỏi dựa trên Fisher Information 
    và difficulty matching. Tất cả tham số có thể tinh chỉnh qua params.

    Args:
        theta (float): khả năng hiện tại.
        gamma (list[float]): hệ số phân biệt.
        beta (list[float]): độ khó.
        unanswered (set[int]): tập câu hỏi chưa trả lời.
        question_meta (dict): metadata (không dùng ở đây).
        difficulty_weight (float): trọng số difficulty trong composite score.
        difficulty_sigma (float): độ rộng Gaussian cho difficulty matching.
        top_k (int): số lượng câu hỏi hàng đầu để random chọn.

    Returns:
        int: ID của câu hỏi tiếp theo.
    """
    # Lấy tham số từ params nếu không được truyền vào
    difficulty_weight = difficulty_weight if difficulty_weight is not None else params.difficulty_weight
    difficulty_sigma = difficulty_sigma if difficulty_sigma is not None else params.difficulty_sigma
    top_k = top_k if top_k is not None else params.top_k

    answered_qids = session.get("answered_questions", [])

    if not answered_qids:
        median_beta = np.median(np.array(beta))
        return min(unanswered, key=lambda q: abs(beta[q] - median_beta))

    scores = {}
    for q in unanswered:
        a = gamma[q]
        b = beta[q]
        p = 1/(1 + np.exp(-a*(theta - b)))
        fisher_info = a**2 * p * (1-p)
        diff_score = np.exp(-((b - theta)**2) / (2 * (difficulty_sigma**2)))
        scores[q] = fisher_info * \
            (difficulty_weight * diff_score + (1 - difficulty_weight))

    sorted_qs = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    candidates = [qid for qid, _ in sorted_qs[:min(top_k, len(sorted_qs))]]
    return random.choice(candidates)


def init_test_session(app):
    from utils.data_loader import load_data, load_irt_params, load_preprocessed_question_metadata, select_anchor_students
    # Load dữ liệu train/test, concept map, metadata, ...
    train_data, test_data, concept_map, metadata = load_data()
    gamma, beta = load_irt_params()
    question_meta, valid_q_ids = load_preprocessed_question_metadata()

    # Lưu các thông tin cần thiết vào app.config
    app.config['QUESTION_META'] = question_meta
    app.config['METADATA'] = metadata
    app.config['CONCEPT_MAP'] = concept_map
    app.config['GAMMA'] = gamma.tolist()
    app.config['BETA'] = beta.tolist()
    app.config['VALID_QUESTION_IDS'] = valid_q_ids

    # Chọn anchor group (toàn bộ học sinh trong train_data)
    anchor_ids = select_anchor_students(train_data)
    app.config['ANCHOR_IDS'] = anchor_ids

    # Load anchor theta đã được tính toán từ compute_anchor_thetas.py
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data", params.data_name)
    try:
        anchor_thetas = np.load(os.path.join(
            data_dir, "anchor_theta.npy"), allow_pickle=True).item()
    except Exception as e:
        anchor_thetas = {}
    app.config['ANCHOR_THETAS'] = np.array(list(anchor_thetas.values()))

    # Khởi tạo session cho phiên thi
    session['student_index'] = 0
    session['current_theta'] = 0.0
    session['current_index'] = 0
    session['score'] = 0
    session['all_questions'] = valid_q_ids
    session['unanswered'] = valid_q_ids[:]
    session['answered_questions'] = []
    session['responses'] = []
    session['a_list'] = []
    session['b_list'] = []


def select_next_question_ccat():
    """
    Sử dụng lớp MCMC_Selection để chọn câu hỏi dựa trên chiến lược CCAT.

    Returns:
        selected_questions: Danh sách các câu hỏi được chọn cho mỗi học sinh test.
        stu_theta: Danh sách theta được cập nhật theo từng bước cho mỗi học sinh.
    """
    # Load dữ liệu cần thiết
    train_data, test_data, concept_map, metadata = load_data()
    gamma, beta = load_irt_params()
    question_meta, valid_q_ids = load_preprocessed_question_metadata()

    # Bạn cần tạo train_label và test_label từ dữ liệu đã được xử lý
    # Giả sử bạn đã có các hàm để tạo chúng, hoặc bạn có thể tái sử dụng từ main.py.
    # Ví dụ (nếu dữ liệu của bạn là dictionary):
    train_label = np.zeros(
        (metadata['num_train_students'], metadata['num_questions'])) - 1
    for stu in range(train_data.num_students):
        for qid, correct in train_data.data[stu].items():
            train_label[stu][qid] = correct
    test_label = np.zeros(
        (metadata['num_test_students'], metadata['num_questions'])) - 1
    for stu in range(test_data.num_students):
        for qid, correct in test_data.data[stu].items():
            test_label[stu][qid] = correct

    # Khởi tạo instance của MCMC_Selection
    selection_instance = MCMC_Selection(
        train_data, test_data, concept_map, train_label, test_label, gamma, beta, params)

    # Gọi hàm get_question của MCMC_Selection để lấy danh sách câu hỏi được chọn cho mỗi học sinh test
    selected_questions, stu_theta = selection_instance.get_question()

    return selected_questions, stu_theta
