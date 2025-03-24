import io
from flask import send_file
import matplotlib.pyplot as plt
import matplotlib
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
from config.session_config import configure_session
from services.question_service import select_next_question, init_test_session
from utils.theta_updater import update_theta_ema, loss_theta, update_theta, update_theta_ccat

# Import Flask-Session
from flask_session import Session

app = Flask(__name__)
configure_session(app)

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


@app.route('/')
def index():
    return render_template("index.html")

# --- Endpoint khởi tạo phiên thi ---


@app.route('/start', methods=['GET'])
def start():
    init_test_session(app)
    return redirect(url_for("question"))

# --- Endpoint hiển thị câu hỏi hiện tại ---


@app.route('/question', methods=['GET'])
def question():
    current_index = session.get('current_index', 0)
    total_questions = 20   # số câu thi cho demo

    if current_index >= total_questions or len(session.get('unanswered', [])) == 0:
        return redirect(url_for("result"))

    current_theta = session.get('current_theta')
    gamma = app.config.get('GAMMA')
    beta = app.config.get('BETA')
    unanswered = session.get('unanswered')
    question_meta = app.config.get('QUESTION_META', {})

    next_qid = select_next_question(
        current_theta, gamma, beta, unanswered, question_meta)
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

    # Lấy thứ hạng hiện tại và tổng số anchor từ session
    current_rank = session.get("current_rank", 0)
    total_anchor = session.get("total_anchor", 0)

    return render_template("question.html",
                           qid=next_qid,
                           image_url=image_url,
                           q_info=q_info,
                           subjects=subject_list,
                           current_index=current_index+1,
                           total=total_questions,
                           current_theta=current_theta,
                           current_rank=current_rank,
                           total_anchor=total_anchor)


# --- Endpoint xử lý đáp án và cập nhật theta ---


@app.route('/submit', methods=['POST'])
def submit():
    user_answer = request.form.get("answer", "").strip().upper()
    current_qid = session.get("current_question")
    question_meta = app.config.get("QUESTION_META", {})
    correct_answer = question_meta.get(
        current_qid, {}).get("correct_answer", "")

    # So sánh đáp án và cập nhật điểm
    is_correct = 1 if user_answer == correct_answer else 0
    session["score"] = session.get("score", 0) + is_correct

    # Cập nhật history trả lời
    answered = session.setdefault("answered_questions", [])
    answered.append(current_qid)
    responses = session.setdefault("responses", [])
    responses.append(is_correct)

    # Cập nhật tham số IRT
    a_list = session.setdefault("a_list", [])
    b_list = session.setdefault("b_list", [])
    gamma = app.config["GAMMA"]
    beta = app.config["BETA"]
    a_list.append(gamma[current_qid])
    b_list.append(beta[current_qid])

    # Xóa câu hỏi đã trả lời khỏi unanswered
    unanswered = session.setdefault("unanswered", [])
    if current_qid in unanswered:
        unanswered.remove(current_qid)

    # Update theta theo CCAT
    current_theta = session.get("current_theta", 0.0)
    anchor_thetas = app.config.get("ANCHOR_THETAS", np.array([]))
    from utils.theta_updater import update_theta_ccat
    new_theta = update_theta_ccat(current_theta, responses, a_list, b_list, anchor_thetas,
                                  lambda_reg=0.005, lambda_ranking=0.5, damping=0.5)
    session["current_theta"] = new_theta

    # Tính thứ hạng (ranking)
    if anchor_thetas.size:
        rank = sum(1 for x in anchor_thetas if x > new_theta) + \
            1  # điểm cao là rank thấp

        session["current_rank"] = rank
        session["total_anchor"] = len(anchor_thetas)
    else:
        session["current_rank"] = 1
        session["total_anchor"] = 1

    # Lưu ranking measure (delta so với mean anchor)
    anchor_mean = np.mean(anchor_thetas) if anchor_thetas.size else 0.0
    delta = new_theta - anchor_mean
    ranking_history = session.setdefault("ranking_history", [])
    ranking_history.append(delta)

    session["current_index"] = session.get("current_index", 0) + 1
    return redirect(url_for("question"))


# --- Endpoint hiển thị kết quả cuối cùng ---


@app.route('/result', methods=['GET'])
def result():
    score = session.get("score", 0)
    total = session.get("current_index", 0)
    final_theta = session.get("current_theta", 0)
    return render_template("result.html", score=score, total=total, final_theta=final_theta)


matplotlib.use('Agg')


@app.route('/ranking_plot.png')
def ranking_plot():
    history = session.get("ranking_history", [])

    if not history:
        history = [0]

    fig, ax = plt.subplots(figsize=(3, 3))  # nhỏ gọn hơn
    x_vals = list(range(1, len(history) + 1))
    ax.plot(x_vals, history, marker='o', linewidth=2, color="#007BFF")

    ax.set_title("Δθ so với anchor", fontsize=8)
    ax.set_xlabel("Câu hỏi", fontsize=7)
    ax.set_ylabel("Δθ", fontsize=7)
    ax.tick_params(axis='both', labelsize=6)

    ax.set_ylim(-3, 3)  # giữ tỉ lệ ổn định cho mọi phiên
    ax.set_xlim(1, max(5, len(history)))  # tránh "đè" vào trục

    plt.tight_layout(pad=1.0)

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100)
    buf.seek(0)
    plt.close(fig)
    return send_file(buf, mimetype='image/png')


if __name__ == '__main__':
    app.run(debug=True)
