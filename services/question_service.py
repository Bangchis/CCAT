from selection_strategy import MCMC_Selection
from utils.data_loader import load_data, load_irt_params, load_preprocessed_question_metadata
import numpy as np
from flask import session
from setting import params
import os
from collections import defaultdict
import random


def select_next_question(theta, gamma, beta, unanswered, question_meta,
                         difficulty_weight=0.5,
                         difficulty_sigma=0.4,
                         top_k=5):
    """
    Chọn ngẫu nhiên một trong top-k câu hỏi dựa trên Fisher Information 
    + bổ sung bonus đa dạng chủ đề. Giữ nguyên logic cũ, chỉ thêm phần "diversity".

    Args:
        theta (float): khả năng hiện tại.
        gamma (list[float]): hệ số phân biệt của các câu hỏi.
        beta (list[float]): độ khó của các câu hỏi.
        unanswered (set[int]): tập câu hỏi chưa trả lời.
        question_meta (dict): metadata câu hỏi (chứa 'subjects').
        difficulty_weight (float): Trọng số độ khó (logic cũ đang bình luận).
        difficulty_sigma (float): Độ rộng Gaussian (logic cũ đang bình luận).
        top_k (int): số lượng câu hỏi hàng đầu để random.

    Returns:
        int: ID của câu hỏi tiếp theo.
    """

    # 1. Lấy tham số từ params nếu hàm không truyền vào
    difficulty_weight = difficulty_weight if difficulty_weight is not None else params.difficulty_weight
    difficulty_sigma = difficulty_sigma if difficulty_sigma is not None else params.difficulty_sigma
    top_k = top_k if top_k is not None else params.top_k
    diversity_w = 0.2  # bạn có thể điều chỉnh

    answered_qids = session.get("answered_questions", [])

    # 2. Nếu người thi chưa trả lời câu nào
    if not answered_qids:
        l = 20  # tùy chọn số lượng câu gần median
        median_beta = np.median(np.array(beta))

        # Sắp xếp unanswered theo độ gần với median_beta (ascending)
        sorted_unanswered = sorted(
            unanswered,
            key=lambda q: abs(beta[q] - median_beta)
        )
        near_median = sorted_unanswered[:l]
        # Chọn ngẫu nhiên 1 trong nhóm "gần median beta"
        return random.choice(near_median)

    # 3. Khi đã trả lời ít nhất 1 câu → tính Fisher + bổ sung đa dạng chủ đề
    #    (Dựa trên logic cũ: fisher_info + random top_k)

    # Tính tần suất chủ đề đã gặp
    subject_freq = {}
    for ans_qid in answered_qids:
        subs = question_meta.get(ans_qid, {}).get('subjects', [])
        for s in subs:
            subject_freq[s] = subject_freq.get(s, 0) + 1

    # Hệ số diversity

    scores = {}
    for q in unanswered:
        a = gamma[q]
        b = beta[q]
        # Fisher Information
        p = 1 / (1 + np.exp(-a * (theta - b)))
        fisher_info = a*a * p * (1-p)

        # Tính "diversity bonus" dựa trên chủ đề
        subs = question_meta.get(q, {}).get('subjects', [])
        if subs:
            div_vals = [1.0 / (1.0 + subject_freq.get(s, 0)) for s in subs]
            div_bonus = sum(div_vals) / len(div_vals)
        else:
            div_bonus = 1.0

        # Điểm cuối = fisher_info * (1 + diversity_w * div_bonus)
        scores[q] = fisher_info * (1 + diversity_w * div_bonus)

    # 4. Lấy top_k câu hỏi có điểm cao nhất, rồi chọn ngẫu nhiên 1
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

    # __file__ ở utils/, parents[2] là project root (CCAT/)
    data_dir = "/mnt/c/Users/Admin/Desktop/code python/CCAT/data/NIPS2020"

    try:
        anchor_dict = np.load(
            os.path.join(data_dir, "anchor_theta.npy"),
            allow_pickle=True
        ).item()
        print("DEBUG: Loaded anchor_theta.npy —", len(anchor_dict), "anchors")
    except Exception as e:
        print("ERROR loading anchor_theta.npy:", e)
        anchor_dict = {}

    app.config['ANCHOR_THETAS'] = np.array(list(anchor_dict.values()))
    # Lấy mảng theta của anchor group
    anchor_thetas = app.config.get('ANCHOR_THETAS', np.array([]))

    # Khởi tạo theta ban đầu
    if anchor_thetas.size > 0:
        # Nếu có anchor, khởi tạo θ bằng trung bình của anchor
        initial_theta = float(np.mean(anchor_thetas))
        print(f"DEBUG: Initial theta set to anchor mean = {initial_theta}")
    else:
        # Nếu không có anchor, dùng giá trị ngẫu nhiên hoặc 0.0
        # Ở đây, ví dụ: random.uniform(-1, 1) trong khoảng [-1, 1]
        import random
        initial_theta = random.uniform(-1.0, 1.0)
        print(
            f"DEBUG: No anchor found. Initial theta (random) = {initial_theta}")

    # Đặt các biến session về trạng thái ban đầu cho một phiên thi mới
    session['student_index'] = 0
    session['current_theta'] = initial_theta
    session['current_index'] = 0
    session['score'] = 0

    # Danh sách câu hỏi có sẵn (valid_q_ids) – đã được load ở trên
    session['all_questions'] = valid_q_ids
    session['unanswered'] = valid_q_ids[:]
    session['answered_questions'] = []
    session['responses'] = []
    session['a_list'] = []
    session['b_list'] = []

    # Reset lịch sử ranking
    session['ranking_history'] = []
    session['current_rank'] = 0
    session['total_anchor'] = len(anchor_thetas)


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
