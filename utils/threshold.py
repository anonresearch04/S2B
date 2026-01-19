import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq


def get_sketch_finalize_threshold(correct_vals, incorrect_vals):
    if len(correct_vals) < 2 or np.all(np.isnan(correct_vals)):
        return 2.0
    if len(incorrect_vals) < 2 or np.all(np.isnan(incorrect_vals)):
        std_ = np.std(correct_vals)
        if std_ > 0.2:
            w = 0
        else:
            w = 3
        threshold = np.mean(correct_vals) - w * std_
        if(threshold < 0):
            threshold = min(correct_vals)
        return threshold

    mu_c, sigma_c = np.mean(correct_vals), np.std(correct_vals)
    mu_i, sigma_i = np.mean(incorrect_vals), np.std(incorrect_vals)

    sigma_c = max(sigma_c, 1e-6)
    sigma_i = max(sigma_i, 1e-6)

    def diff(x):
        return norm.pdf(x, mu_c, sigma_c) - norm.pdf(x, mu_i, sigma_i)

    threshold = mu_c - 3 * sigma_c
    if(threshold < 0):
        threshold = min(correct_vals)
    try:
        lower, upper = sorted([mu_c, mu_i])
        threshold = brentq(diff, lower, upper)
    except Exception as e:
        pass
    return threshold


def get_sketch_ood_threshold(correct_vals, incorrect_vals, sigma=3):
    threshold = 0
    if len(incorrect_vals) != 0:
        mean_score = np.mean(incorrect_vals)
        std_score = np.std(incorrect_vals)
        threshold = mean_score - sigma * std_score
    else:
        if(len(correct_vals) != 0):
            threshold = min(correct_vals)
    return threshold

