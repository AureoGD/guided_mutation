import torch
from torch.utils.data import Dataset


class ValueDataset(Dataset):

    def __init__(self, X, y):
        """
        X: numpy array (N, window, obs_dim + 1)
        y: numpy array (N,) -> Delta Recompensa
        """
        # Armazenamos como numpy para economizar memória inicial
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Conversão para Tensor acontece sob demanda
        # Usamos float32 explicitamente (padrão para redes neurais)
        sample_x = torch.from_numpy(self.X[idx]).float()
        sample_y = torch.tensor(self.y[idx]).float()

        return sample_x, sample_y
