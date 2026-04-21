import numpy as np


def normalize_state(state):
    state_min = np.array([-2.5, -2.5, -10.0, -10.0, -6.2831855, -10.0, 0.0, 0.0])
    state_max = np.array([2.5, 2.5, 10.0, 10.0, 6.2831855, 10.0, 1.0, 1.0])

    # Previne divisão por zero caso min == max
    range_val = state_max - state_min
    range_val[range_val == 0] = 1e-8

    # Normalização Min-Max escalonada para [-1, 1]
    normalized = 2.0 * (state - state_min) / range_val - 1.0

    # Clip de segurança para garantir que outliers não quebrem o range
    return np.clip(normalized, -1.0, 1.0)
