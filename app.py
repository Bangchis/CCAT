import os
import json
import numpy as np
import torch
import random
import pandas as pd
import scipy.optimize
import ast  # dùng để chuyển chuỗi "[3, 71, ...]" thành list
from flask import Flask, request, session, render_template, redirect, url_for
from dataset import AdapTestDataset
from setting import params  # Sử dụng các tham số từ setting.py
from selection_strategy import MCMC_Selection

# Import Flask-Session
from flask_session import Session

app = Flask(__name__)
app.secret_key = 'demo_secret_key'

# Cấu hình session để lưu ở server (filesystem)
app.config["SESSION_TYPE"] = "filesystem"
app.config["SESSION_FILE_DIR"] = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "flask_session")
app.config["SESSION_PERMANENT"] = False
Session(app)

# --- Hàm load dữ liệu cơ bản từ NIPS2020 ---


def load_data():
    # Sử dụng __file__ để xác định thư mục gốc của dự án (CCAT)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    # Giả sử dữ liệu nằm trong thư mục con "data" của thư mục dự án
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

# --- Hàm load tham số IRT (gamma, beta) ---


def load_irt_params():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data", params.data_name)
    gamma = np.load(os.path.join(data_dir, "alpha.npy"))
    beta = np.load(os.path.join(data_dir, "beta.npy"))
    return gamma, beta

# --- Hàm load thông tin câu hỏi từ file preprocessed_questions.csv ---


def load_preprocessed_question_metadata():
    base_dir = os.path.dirname(os.path.abspath(__file__))
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

# --- Hàm Loss dùng Binary Cross-Entropy (BCE) để cập nhật theta ---


def loss_theta(theta, y, a, b, lambda_reg=0.05):
    # theta: scalar; y, a, b: vector (numpy arrays)
    p = 1 / (1 + np.exp(-a * (theta - b)))
    eps = 1e-6
    ce_loss = - (y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))
    return ce_loss.sum() + lambda_reg * theta**2

# --- Hàm cập nhật theta theo phương pháp EMA (dựa trên phản hồi của câu hỏi hiện tại) ---


def update_theta_ema(current_theta, response, a, b, alpha=0.1):
    """
    Cập nhật theta chỉ dựa trên câu hỏi hiện tại.
      - response: 1 nếu đúng, 0 nếu sai.
      - a, b: tham số của câu hỏi hiện tại.
      - alpha: hệ số EMA (learning rate) điều chỉnh mức phản ứng với phản hồi mới.
    """
    # Tính xác suất dự đoán theo mô hình IRT với sigmoid
    p = 1 / (1 + np.exp(-a * (current_theta - b)))
    # Ước lượng "theta hiện tại" dựa trên động lực gradient:
    theta_current = current_theta + a * (response - p)
    # Cập nhật theo EMA: giao động mới chủ yếu dựa trên thông tin của câu hỏi hiện tại
    new_theta = (1 - alpha) * current_theta + alpha * theta_current
    new_theta = max(min(new_theta, 3), -3)
    return new_theta

# --- Hàm cập nhật theta với cơ chế damping (truy thu toàn bộ phản hồi, vẫn giữ logic cũ) ---


def update_theta(current_theta, responses, a_list, b_list):
    if len(responses) == 0:
        return current_theta
    num_responses = len(responses)
    if num_responses < 4:
        damping = 0.4
    elif num_responses < 14:
        damping = 0.6
    else:
        damping = 0.8
    responses_arr = np.array(responses)
    a_arr = np.array(a_list)
    b_arr = np.array(b_list)
    res = scipy.optimize.minimize(
        loss_theta,
        current_theta,
        args=(responses_arr, a_arr, b_arr, 0.05),
        bounds=[(-3, 3)]
    )
    optimal_theta = res.x[0]
    new_theta = current_theta + damping * (optimal_theta - current_theta)
    new_theta = max(min(new_theta, 3), -3)
    return new_theta

# --- Hàm khởi tạo phiên thi cho 1 học sinh ---


def init_test_session():
    train_data, test_data, concept_map, metadata = load_data()
    gamma, beta = load_irt_params()
    question_meta, valid_q_ids = load_preprocessed_question_metadata()
    app.config['QUESTION_META'] = question_meta
    app.config['METADATA'] = metadata
    app.config['CONCEPT_MAP'] = concept_map
    app.config['GAMMA'] = gamma.tolist()
    app.config['BETA'] = beta.tolist()
    app.config['VALID_QUESTION_IDS'] = valid_q_ids
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

# --- Endpoint trang chủ ---


@app.route('/')
def index():
    return render_template("index.html")

# --- Endpoint khởi tạo phiên thi ---


@app.route('/start', methods=['GET'])
def start():
    init_test_session()
    return redirect(url_for("question"))

# --- Endpoint hiển thị câu hỏi hiện tại ---


@app.route('/question', methods=['GET'])
def question():
    current_index = session.get('current_index', 0)
    total_questions = 40  # số câu thi cho demo
    if current_index >= total_questions or len(session.get('unanswered', [])) == 0:
        return redirect(url_for("result"))
    current_theta = session.get('current_theta')
    gamma = app.config.get('GAMMA')
    beta = app.config.get('BETA')
    unanswered = session.get('unanswered')
    question_meta = app.config.get('QUESTION_META', {})
    valid_q_ids = app.config.get('VALID_QUESTION_IDS', [])
    # Hàm chọn câu hỏi kết hợp nhiều tiêu chí: Fisher Information, diversity bonus và help bonus

    def select_next_question(theta, gamma, beta, unanswered):
        diversity_weight = 0.0
        help_bonus_weight = 0.0
        answered_qids = session.get("answered_questions", [])
        if not answered_qids:
            beta_array = np.array(beta)
            median_beta = np.median(beta_array)
            best_qid = min(unanswered, key=lambda qid: abs(
                beta[qid] - median_beta))
            print("DEBUG: First question selected based on median beta:", best_qid)
            return best_qid
        fisher_info = {}
        subject_freq = {}
        for qid in answered_qids:
            subjects = question_meta.get(qid, {}).get("subjects", [])
            for subj in subjects:
                subject_freq[subj] = subject_freq.get(subj, 0) + 1
        for qid in unanswered:
            a = gamma[qid]
            b = beta[qid]
            P = 1 / (1 + np.exp(-a * (theta - b)))
            I = a**2 * P * (1 - P)
            bonus = 0.0
            candidate_subjects = question_meta.get(qid, {}).get("subjects", [])
            if candidate_subjects:
                bonus_vals = []
                for subj in candidate_subjects:
                    freq = subject_freq.get(subj, 0)
                    bonus_vals.append(1 / (1 + freq))
                diversity_bonus = np.mean(bonus_vals)
                bonus += diversity_weight * diversity_bonus
            if theta < -3 and b < theta:
                bonus += help_bonus_weight
            fisher_info[qid] = I * (1 + bonus)
        max_I = max(fisher_info.values())
        if theta < -2 and max_I < 0.1:
            fallback_qid = min(unanswered, key=lambda qid: beta[qid])
            print(
                "DEBUG: Fallback question selected (due to low theta and low Fisher Info):", fallback_qid)
            return fallback_qid
        return max(fisher_info, key=fisher_info.get)
    next_qid = select_next_question(current_theta, gamma, beta, unanswered)
    session['current_question'] = next_qid
    q_info = question_meta.get(next_qid, {})
    print("DEBUG: Selected Question (New ID):", next_qid)
    print("DEBUG: Correct Answer:", q_info.get("correct_answer", ""))
    print("DEBUG: Subjects:", q_info.get("subjects", ""))
    old_qid = q_info.get("old_question_id", next_qid)
    image_url = url_for(
        'static', filename=f'images/{params.data_name}/{old_qid}.jpg')
    subjects_raw = q_info.get('subjects', "")
    if isinstance(subjects_raw, str):
        subject_list = [s.strip()
                        for s in subjects_raw.split(',') if s.strip()]
    else:
        subject_list = subjects_raw
    return render_template("question.html",
                           qid=next_qid,
                           image_url=image_url,
                           q_info=q_info,
                           subjects=subject_list,
                           current_index=current_index+1,
                           total=total_questions,
                           current_theta=current_theta)

# --- Endpoint xử lý đáp án và cập nhật theta ---


@app.route('/submit', methods=['POST'])
def submit():
    user_answer = request.form.get("answer", "").strip().upper()
    current_qid = session.get("current_question")
    question_meta = app.config.get("QUESTION_META", {})
    correct_answer = question_meta.get(
        current_qid, {}).get("correct_answer", "")
    print("DEBUG: User Answer:", user_answer)
    print("DEBUG: Correct Answer:", correct_answer)
    is_correct = 1 if user_answer == correct_answer else 0
    score = session.get("score", 0)
    if is_correct:
        score += 1
    session["score"] = score
    answered = session.get("answered_questions", [])
    responses = session.get("responses", [])
    a_list = session.get("a_list", [])
    b_list = session.get("b_list", [])
    answered.append(current_qid)
    responses.append(is_correct)
    gamma = app.config.get("GAMMA")
    beta = app.config.get("BETA")
    a_list.append(gamma[current_qid])
    b_list.append(beta[current_qid])
    session["answered_questions"] = answered
    session["responses"] = responses
    session["a_list"] = a_list
    session["b_list"] = b_list
    unanswered = session.get("unanswered", [])
    if current_qid in unanswered:
        unanswered.remove(current_qid)
    session["unanswered"] = unanswered

    # Cập nhật theta sử dụng phương pháp EMA chỉ dựa trên phản hồi của câu hỏi hiện tại
    current_theta = session.get("current_theta")
    a_current = gamma[current_qid]
    b_current = beta[current_qid]
    new_theta = update_theta_ema(
        current_theta, is_correct, a_current, b_current, alpha=0.25)
    session["current_theta"] = new_theta

    current_index = session.get("current_index", 0)
    session["current_index"] = current_index + 1
    return redirect(url_for("question"))

# --- Endpoint hiển thị kết quả cuối cùng ---


@app.route('/result', methods=['GET'])
def result():
    score = session.get("score", 0)
    total = session.get("current_index", 0)
    final_theta = session.get("current_theta", 0)
    return render_template("result.html", score=score, total=total, final_theta=final_theta)


if __name__ == '__main__':
    app.run(debug=True)
