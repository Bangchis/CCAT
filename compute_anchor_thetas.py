import numpy as np
import pandas as pd
import json
import scipy.optimize
import os
from collections import defaultdict
from setting import params


def parse_train_data(train_csv):
    """
    Đọc file train_triples.csv và chuyển sang dạng dictionary:
    {student_id: {question_id: correct, ...}, ...}
    """
    df = pd.read_csv(train_csv)
    stu_data = defaultdict(dict)
    for _, row in df.iterrows():
        stu_data[int(row['student_id'])][int(
            row['question_id'])] = int(row['correct'])
    return stu_data


def likelihood(theta, y, a, b, lambda_reg=0.4):
    """
    Hàm likelihood đơn giản cho IRT.
    """
    # Tính xác suất trả lời đúng cho từng item
    exponent = -a * (theta - b)
    exponent = np.clip(exponent, -50, 50)  # tùy chỉnh khoảng này phù hợp
    p = 1 / (1 + np.exp(exponent))

    # Tính log likelihood, tránh log(0) bằng cách thêm một số nhỏ eps
    eps = 1e-2
    log_likelihood = (y * np.log(p + eps) + (1 - y)
                      * np.log(1 - p + eps)).sum()

    # Thêm điều khoản penalty để hạn chế giá trị θ lan man
    penalty = lambda_reg * theta**2
    return - (log_likelihood - penalty)


def compute_anchor_thetas(train_csv, gamma, beta):
    """
    Ước lượng theta cho từng học sinh trong tập train.

    Args:
        train_csv: Đường dẫn đến file train_triples.csv.
        gamma: Mảng tham số phân biệt (được load từ alpha.npy).
        beta: Mảng tham số độ khó (được load từ beta.npy).

    Returns:
        anchor_thetas: Dictionary {student_id: theta}
    """
    stu_data = parse_train_data(train_csv)
    anchor_thetas = {}
    for stu_id, responses in stu_data.items():
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
        # Giới hạn theta trong khoảng [-4,4] (hoặc theo yêu cầu)
        theta = np.clip(theta, -3, 3)
        anchor_thetas[stu_id] = theta
    return anchor_thetas


if __name__ == '__main__':
    # Xác định thư mục dự án
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data", params.data_name)
    train_csv = os.path.join(data_dir, "train_triples.csv")

    # Load tham số IRT
    gamma = np.load(os.path.join(data_dir, "alpha.npy"))
    beta = np.load(os.path.join(data_dir, "beta.npy"))

    # Tính theta cho anchor group
    anchor_thetas = compute_anchor_thetas(train_csv, gamma, beta)
    print("Anchor thetas computed for {} students.".format(len(anchor_thetas)))
    print("Anchor thetas range:")

    print("Min:", min(anchor_thetas.values()))
    print("Max:", max(anchor_thetas.values()))
    print("Mean:", np.mean(list(anchor_thetas.values())))
    print("Std:", np.std(list(anchor_thetas.values())))

    # In ra các giá trị theta
    # Lưu kết quả ra file, ví dụ anchor_theta.npy
    np.save(os.path.join(data_dir, "anchor_theta.npy"), anchor_thetas)
