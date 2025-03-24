# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import json
import math
from numba import jit
from tqdm import tqdm
import os

from dataset import Dataset, TrainDataset, AdapTestDataset
from setting import params

# ---------------------
# 1. Load data (tối ưu sử dụng list comprehension và chuyển sang numpy array một lần)
# ---------------------


def load_data(train_file, metadata_file):
    triplets = pd.read_csv(
        train_file, encoding='utf-8').to_records(index=False)
    metadata = json.load(open(metadata_file, 'r'))
    train_data = AdapTestDataset(
        triplets, metadata['num_train_students'], metadata['num_questions'])

    data_j = []
    data_k = []
    data_y = []

    for j, items in train_data.data.items():
        for k, y in items.items():
            data_j.append(j)
            data_k.append(k)
            data_y.append(y)

    data_j = np.array(data_j, dtype=np.int32)
    data_k = np.array(data_k, dtype=np.int32)
    data_y = np.array(data_y, dtype=np.int32)

    data_dict = {
        'J': metadata['num_train_students'],
        'K': metadata['num_questions'],
        'N': len(data_y),
        'j': data_j,
        'k': data_k,
        'y': data_y
    }

    return data_dict, metadata

# ---------------------
# 2. Định nghĩa hàm tính log-posterior với Numba để tăng tốc
# ---------------------


@jit(nopython=True)
def log_prior(mu_beta, sigma_beta, sigma_gamma, alpha, beta, gamma):
    if sigma_beta <= 0 or sigma_gamma <= 0 or np.any(gamma <= 0):
        return -np.inf

    log_prior_mu_beta = -np.log(5 * math.pi * (1 + (mu_beta/5)**2))
    log_prior_sigma_beta = -np.log(5 * math.pi * (1 + (sigma_beta/5)**2))
    log_prior_sigma_gamma = -np.log(5 * math.pi * (1 + (sigma_gamma/5)**2))
    log_prior_alpha = -0.5 * np.sum(alpha**2)
    log_prior_beta = -0.5 * \
        np.sum((beta**2) / (sigma_beta**2)) - len(beta) * np.log(sigma_beta)
    log_prior_gamma = -0.5 * np.sum((np.log(gamma)**2) / (sigma_gamma**2)) - np.sum(
        np.log(gamma)) - len(gamma) * np.log(sigma_gamma)

    return log_prior_mu_beta + log_prior_sigma_beta + log_prior_sigma_gamma + log_prior_alpha + log_prior_beta + log_prior_gamma


@jit(nopython=True)
def log_likelihood(alpha, beta, gamma, mu_beta, j_idx, k_idx, y):
    logits = gamma[k_idx] * (alpha[j_idx] - (beta[k_idx] + mu_beta))
    pos_part = y * (-np.log1p(np.exp(-logits)))
    neg_part = (1 - y) * (-np.log1p(np.exp(logits)))
    return np.sum(pos_part + neg_part)


@jit(nopython=True)
def log_posterior(mu_beta, sigma_beta, sigma_gamma, alpha, beta, gamma, j_idx, k_idx, y):
    lp = log_prior(mu_beta, sigma_beta, sigma_gamma, alpha, beta, gamma)
    if np.isinf(lp):
        return -np.inf
    ll = log_likelihood(alpha, beta, gamma, mu_beta, j_idx, k_idx, y)
    return lp + ll

# ---------------------
# 3. Hàm MCMC sampler tối ưu với block update và cập nhật hiệu chỉnh delta
# ---------------------


def mcmc_sampler(initial_params, data, num_iterations, proposal_scales, thinning=10, burn_in_pct=0.2):
    J = data['J']
    K = data['K']
    j_idx = data['j']
    k_idx = data['k']
    y = data['y']

    # Tiền xử lý: Tạo danh sách các chỉ số (index mapping) cho từng học sinh và câu hỏi
    student_idx_map = [np.where(j_idx == j)[0] for j in range(J)]
    question_idx_map = [np.where(k_idx == k)[0] for k in range(K)]

    # Khởi tạo các tham số hiện tại
    current = {
        'mu_beta': initial_params['mu_beta'],
        'sigma_beta': initial_params['sigma_beta'],
        'sigma_gamma': initial_params['sigma_gamma'],
        'alpha': initial_params['alpha'].copy(),
        'beta': initial_params['beta'].copy(),
        'gamma': initial_params['gamma'].copy()
    }

    current_log_post = log_posterior(
        current['mu_beta'], current['sigma_beta'], current['sigma_gamma'],
        current['alpha'], current['beta'], current['gamma'],
        j_idx, k_idx, y
    )

    num_samples = num_iterations // thinning
    samples = {
        'mu_beta': np.zeros(num_samples),
        'sigma_beta': np.zeros(num_samples),
        'sigma_gamma': np.zeros(num_samples),
        'alpha': np.zeros((num_samples, J)),
        'beta': np.zeros((num_samples, K)),
        'gamma': np.zeros((num_samples, K))
    }

    # Đếm số lần chấp nhận
    accept_count = {key: 0 for key in [
        'mu_beta', 'sigma_beta', 'sigma_gamma', 'alpha', 'beta', 'gamma']}
    total_count = {key: 0 for key in [
        'mu_beta', 'sigma_beta', 'sigma_gamma', 'alpha', 'beta', 'gamma']}

    block_size_alpha = min(10, J)
    block_size_beta = min(10, K)
    block_size_gamma = min(10, K)

    for it in tqdm(range(num_iterations)):
        # --- Cập nhật mu_beta (scalar) ---
        total_count['mu_beta'] += 1
        prop_mu_beta = current['mu_beta'] + \
            proposal_scales['mu_beta'] * np.random.randn()
        log_post_prop = log_posterior(
            prop_mu_beta, current['sigma_beta'], current['sigma_gamma'],
            current['alpha'], current['beta'], current['gamma'],
            j_idx, k_idx, y
        )
        if np.log(np.random.rand()) < log_post_prop - current_log_post:
            current['mu_beta'] = prop_mu_beta
            current_log_post = log_post_prop
            accept_count['mu_beta'] += 1

        # --- Cập nhật sigma_beta (scalar, >0) ---
        total_count['sigma_beta'] += 1
        prop_sigma_beta = current['sigma_beta'] * \
            np.exp(proposal_scales['sigma_beta'] * np.random.randn())
        log_post_prop = log_posterior(
            current['mu_beta'], prop_sigma_beta, current['sigma_gamma'],
            current['alpha'], current['beta'], current['gamma'],
            j_idx, k_idx, y
        )
        log_accept_ratio = log_post_prop - current_log_post + \
            np.log(prop_sigma_beta/current['sigma_beta'])
        if np.log(np.random.rand()) < log_accept_ratio:
            current['sigma_beta'] = prop_sigma_beta
            current_log_post = log_post_prop
            accept_count['sigma_beta'] += 1

        # --- Cập nhật sigma_gamma (scalar, >0) ---
        total_count['sigma_gamma'] += 1
        prop_sigma_gamma = current['sigma_gamma'] * \
            np.exp(proposal_scales['sigma_gamma'] * np.random.randn())
        log_post_prop = log_posterior(
            current['mu_beta'], current['sigma_beta'], prop_sigma_gamma,
            current['alpha'], current['beta'], current['gamma'],
            j_idx, k_idx, y
        )
        log_accept_ratio = log_post_prop - current_log_post + \
            np.log(prop_sigma_gamma/current['sigma_gamma'])
        if np.log(np.random.rand()) < log_accept_ratio:
            current['sigma_gamma'] = prop_sigma_gamma
            current_log_post = log_post_prop
            accept_count['sigma_gamma'] += 1

        # --- Cập nhật vector alpha theo block ---
        for start_idx in range(0, J, block_size_alpha):
            end_idx = min(start_idx + block_size_alpha, J)
            block_indices = np.arange(start_idx, end_idx)
            # Lấy các mẫu bị ảnh hưởng: các phần tử dữ liệu liên quan đến các học sinh trong block
            affected = np.concatenate([student_idx_map[j]
                                      for j in block_indices])
            # Tính lại log-likelihood trên vùng bị ảnh hưởng
            ll_old = log_likelihood(current['alpha'], current['beta'], current['gamma'], current['mu_beta'],
                                    j_idx[affected], k_idx[affected], y[affected])
            # Tính log-prior cho block alpha (prior N(0,1))
            prior_old = -0.5 * np.sum(current['alpha'][block_indices]**2)
            # Đề xuất cập nhật cho block alpha
            prop_alpha = current['alpha'].copy()
            prop_alpha[block_indices] += proposal_scales['alpha'] * \
                np.random.randn(end_idx - start_idx)
            ll_new = log_likelihood(prop_alpha, current['beta'], current['gamma'], current['mu_beta'],
                                    j_idx[affected], k_idx[affected], y[affected])
            prior_new = -0.5 * np.sum(prop_alpha[block_indices]**2)
            delta = (ll_new - ll_old) + (prior_new - prior_old)
            total_count['alpha'] += 1
            if np.log(np.random.rand()) < delta:
                current['alpha'][block_indices] = prop_alpha[block_indices]
                current_log_post += delta
                accept_count['alpha'] += 1

        # --- Cập nhật vector beta theo block ---
        for start_idx in range(0, K, block_size_beta):
            end_idx = min(start_idx + block_size_beta, K)
            block_indices = np.arange(start_idx, end_idx)
            affected = np.concatenate([question_idx_map[k]
                                      for k in block_indices])
            ll_old = log_likelihood(current['alpha'], current['beta'], current['gamma'], current['mu_beta'],
                                    j_idx[affected], k_idx[affected], y[affected])
            prior_old = -0.5 * np.sum((current['beta'][block_indices]**2) / (
                current['sigma_beta']**2)) - (end_idx - start_idx)*np.log(current['sigma_beta'])
            prop_beta = current['beta'].copy()
            prop_beta[block_indices] += proposal_scales['beta'] * \
                np.random.randn(end_idx - start_idx)
            ll_new = log_likelihood(current['alpha'], prop_beta, current['gamma'], current['mu_beta'],
                                    j_idx[affected], k_idx[affected], y[affected])
            prior_new = -0.5 * np.sum((prop_beta[block_indices]**2) / (
                current['sigma_beta']**2)) - (end_idx - start_idx)*np.log(current['sigma_beta'])
            delta = (ll_new - ll_old) + (prior_new - prior_old)
            total_count['beta'] += 1
            if np.log(np.random.rand()) < delta:
                current['beta'][block_indices] = prop_beta[block_indices]
                current_log_post += delta
                accept_count['beta'] += 1

        # --- Cập nhật vector gamma theo block (sử dụng log-scale proposal) ---
        for start_idx in range(0, K, block_size_gamma):
            end_idx = min(start_idx + block_size_gamma, K)
            block_indices = np.arange(start_idx, end_idx)
            affected = np.concatenate([question_idx_map[k]
                                      for k in block_indices])
            ll_old = log_likelihood(current['alpha'], current['beta'], current['gamma'], current['mu_beta'],
                                    j_idx[affected], k_idx[affected], y[affected])
            # Prior cho gamma: LogNormal(0, sigma_gamma)
            prior_old = -0.5 * np.sum((np.log(current['gamma'][block_indices])**2) / (current['sigma_gamma']**2)) - np.sum(
                np.log(current['gamma'][block_indices])) - (end_idx - start_idx)*np.log(current['sigma_gamma'])
            prop_gamma = current['gamma'].copy()
            old_log_gamma = np.log(current['gamma'][block_indices])
            new_log_gamma = old_log_gamma + \
                proposal_scales['gamma'] * np.random.randn(end_idx - start_idx)
            prop_gamma[block_indices] = np.exp(new_log_gamma)
            ll_new = log_likelihood(current['alpha'], current['beta'], prop_gamma, current['mu_beta'],
                                    j_idx[affected], k_idx[affected], y[affected])
            prior_new = -0.5 * np.sum((new_log_gamma**2) / (current['sigma_gamma']**2)) - np.sum(
                new_log_gamma) - (end_idx - start_idx)*np.log(current['sigma_gamma'])
            # Điều chỉnh Jacobian cho chuyển đổi log
            delta_jacobian = np.sum(new_log_gamma - old_log_gamma)
            delta = (ll_new - ll_old) + \
                (prior_new - prior_old) + delta_jacobian
            total_count['gamma'] += 1
            if np.log(np.random.rand()) < delta:
                current['gamma'][block_indices] = prop_gamma[block_indices]
                current_log_post += delta
                accept_count['gamma'] += 1

        # Lưu mẫu sau mỗi lần thinning
        if (it + 1) % thinning == 0:
            idx = (it + 1) // thinning - 1
            samples['mu_beta'][idx] = current['mu_beta']
            samples['sigma_beta'][idx] = current['sigma_beta']
            samples['sigma_gamma'][idx] = current['sigma_gamma']
            samples['alpha'][idx] = current['alpha'].copy()
            samples['beta'][idx] = current['beta'].copy()
            samples['gamma'][idx] = current['gamma'].copy()

    acceptance_rates = {k: (accept_count[k] / total_count[k])
                        if total_count[k] > 0 else 0 for k in accept_count.keys()}

    return samples, acceptance_rates, int(burn_in_pct * num_samples)

# ---------------------
# 4. Hàm tính các ước lượng sau burn-in
# ---------------------


def compute_posterior_estimates(samples, burn_in):
    estimates = {}
    for key, value in samples.items():
        if isinstance(value, np.ndarray):
            if value.ndim > 1:
                estimates[key] = np.mean(value[burn_in:], axis=0)
            else:
                estimates[key] = np.mean(value[burn_in:])
    return estimates

# ---------------------
# 5. Hàm chính chạy toàn bộ quá trình
# ---------------------


def run_irt_mcmc(train_file, metadata_file, output_dir, num_iterations=10000, thinning=10):
    print("Đang tải dữ liệu...")
    data_dict, metadata = load_data(train_file, metadata_file)
    J = data_dict['J']
    K = data_dict['K']
    print(
        f"Dữ liệu có {J} học sinh và {K} câu hỏi với {data_dict['N']} quan sát")

    initial_params = {
        'mu_beta': 0.0,
        'sigma_beta': 1.0,
        'sigma_gamma': 1.0,
        'alpha': np.zeros(J),
        'beta': np.zeros(K),
        'gamma': np.ones(K)
    }

    proposal_scales = {
        'mu_beta': 0.05,
        'sigma_beta': 0.05,
        'sigma_gamma': 0.05,
        'alpha': 0.1,
        'beta': 0.1,
        'gamma': 0.1
    }

    print("Bắt đầu chạy MCMC...")
    samples, acceptance_rates, burn_in = mcmc_sampler(
        initial_params, data_dict, num_iterations,
        proposal_scales, thinning=thinning
    )

    print("Tính toán các ước lượng...")
    estimates = compute_posterior_estimates(samples, burn_in)

    print("Lưu các ước lượng...")
    os.makedirs(output_dir, exist_ok=True)
    np.save(f"{output_dir}/beta.npy", estimates['beta'] + estimates['mu_beta'])
    np.save(f"{output_dir}/alpha.npy", estimates['gamma'])

    print("Tỉ lệ chấp nhận:")
    for key, rate in acceptance_rates.items():
        print(f"{key}: {rate:.4f}")

    return samples, estimates, acceptance_rates


# ---------------------
# Chạy trực tiếp khi file được thực thi
# ---------------------
if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data", "NIPS2020")
    train_file = os.path.join(data_dir, "train_triples.csv")
    metadata_file = os.path.join(data_dir, "metadata.json")
    output_dir = params.data_name

    # Sử dụng ít iteration hơn cho mục đích demo
    samples, estimates, acceptance_rates = run_irt_mcmc(
        train_file, metadata_file, output_dir,
        num_iterations=10000, thinning=20
    )

    print("Hoàn thành quá trình MCMC!")
