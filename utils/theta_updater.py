import numpy as np
import scipy.optimize


def loss_theta(theta, y, a, b, lambda_reg=0.05):
    """
    Tính Binary Cross-Entropy Loss cho mô hình IRT với regularization.

    Args:
        theta: float, khả năng cần ước lượng.
        y: numpy.array, phản hồi của học sinh (0/1).
        a: numpy.array, tham số phân biệt của các câu hỏi đã trả lời.
        b: numpy.array, tham số độ khó của các câu hỏi đã trả lời.
        lambda_reg: float, hệ số regularization.

    Returns:
        Tổng loss.
    """
    eps = 1e-6
    p = 1 / (1 + np.exp(-a * (theta - b)))
    ce_loss = - (y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))
    return ce_loss.sum() + lambda_reg * theta**2


def pairwise_ranking_loss(theta, anchor_thetas, a_arr, b_arr, responses, margin=0.3):
    """
    Hinge‑style loss: phạt mỗi anchor nếu thứ tự giữa người thi và anchor sai lệch.
    """
    loss = 0.0
    # Tính xác suất dự đoán của người thi
    p_user = 1/(1+np.exp(-a_arr*(theta - b_arr)))
    for anchor_theta in anchor_thetas:
        # Tính xác suất dự đoán của anchor cho cùng bộ câu hỏi
        p_anchor = 1/(1+np.exp(-a_arr*(anchor_theta - b_arr)))
        # Nếu người thi trả lời đúng, phạt khi p_user < p_anchor
        loss += np.sum(np.maximum(0, margin + p_anchor - p_user)
                       * (responses == 1))
        # Nếu người thi trả lời sai, phạt khi p_user > p_anchor
        loss += np.sum(np.maximum(0, margin + p_user - p_anchor)
                       * (responses == 0))
    return loss


def update_theta_ema(current_theta, response, a, b, alpha=0.1):
    """
    Cập nhật theta dựa trên câu hỏi hiện tại theo cách EMA (Exponential Moving Average).
    """
    p = 1 / (1 + np.exp(-a * (current_theta - b)))
    theta_current = current_theta + a * (response - p)
    new_theta = (1 - alpha) * current_theta + alpha * theta_current
    new_theta = max(min(new_theta, 3), -3)
    return new_theta


def update_theta(current_theta, responses, a_list, b_list):
    """
    Cập nhật theta dựa trên tập hợp phản hồi của người thi với hệ số damping.
    """
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


def update_theta_ccat(current_theta, responses, a_list, b_list, anchor_thetas,
                      lambda_reg=0.01, lambda_ranking=0.1, damping=0.4):
    responses_arr = np.array(responses)
    a_arr = np.array(a_list)
    b_arr = np.array(b_list)

    def composite_loss(theta):
        return loss_theta(theta, responses_arr, a_arr, b_arr, lambda_reg) + \
            lambda_ranking * \
            pairwise_ranking_loss(theta, anchor_thetas,
                                  a_arr, b_arr, responses_arr)

    res = scipy.optimize.minimize(
        composite_loss, current_theta, bounds=[(-3, 3)])
    optimal_theta = res.x[0]
    new_theta = current_theta + damping * (optimal_theta - current_theta)
    return float(np.clip(new_theta, -3, 3))
