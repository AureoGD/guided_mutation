import numpy as np


def build_dataset_from_memory(memory, window_size=5, val_split=0.2, predict_delta=True):
    X = []
    y = []
    eps = 1e-6

    # Supondo que seu buffer armazene (s, a, r, s_next, done)
    transitions = list(memory.buffer)

    if len(transitions) < window_size + 1:
        return ([], []), ([], [])

    for t in range(window_size, len(transitions)):

        window_slice = transitions[t - window_size:t + 1]
        if any(trans[4] for trans in window_slice):
            continue

        window_data = []
        for i in range(t - window_size, t):
            s = transitions[i][0]
            a = transitions[i][1]
            window_data.append(np.concatenate([s, np.array([a])]))

        if predict_delta:
            reward_t = transitions[t][2]
            reward_prev = transitions[t - window_size][2]

            # target = (reward_t - reward_prev) / (abs(reward_prev) + eps)

            diff = reward_t - reward_prev
            target = np.sign(diff) * np.log1p(np.abs(diff))
        else:
            target = transitions[t][2]

        X.append(window_data)
        y.append(target)

    X = np.array(X)
    y = np.array(y)

    if len(X) == 0:
        return ([], []), ([], [])

    # Split de treino e validação
    idx = np.random.permutation(len(X))
    split = int(len(X) * (1 - val_split))

    train_idx, val_idx = idx[:split], idx[split:]
    return (X[train_idx], y[train_idx]), (X[val_idx], y[val_idx])
