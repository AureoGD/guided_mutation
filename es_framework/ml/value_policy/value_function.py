import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

from guided_mutation.es_framework.ml.value_policy.model import ValueCNN
from guided_mutation.es_framework.ml.value_policy.trainer import ValueTrainer
from guided_mutation.es_framework.ml.value_policy.dataset import ValueDataset
from guided_mutation.es_framework.ml.value_policy.dataset_builder import build_dataset_from_memory


class ValueFunction:

    def __init__(self, config, device="cpu"):
        """
        config: Dicionário contendo env_spec, window_size, lr, epochs, loss_threshold, etc.
        """
        self.config = config["v_policy"]

        self.window_size = config.get("window_size", 5)
        self.device = torch.device(device)

        # Hiperparâmetros de treino extraídos do config
        self.lr = config.get("lr", 1e-3)
        self.epochs = config.get("epochs", 25)
        self.loss_threshold = config.get("loss_threshold", 1e-4)
        self.batch_size = config.get("batch_size", 32)

        # Entrada: s + a (dimensão total das features por timestep)
        env_spec = config["env_spec"]
        input_dim = env_spec.obs_dim + 1

        # ----------------------------------------
        # MODELO
        # ----------------------------------------
        self.model = ValueCNN(obs_dim=input_dim, window_size=self.window_size).to(self.device)

        # Salva o device no modelo para facilitar o acesso no Trainer
        self.model.device = self.device
        self.trainer = ValueTrainer(self.model, lr=self.lr)

    def train(self, memory):
        # Agora build_dataset_from_memory usa a flag para delta_r
        (train_X, train_y), (val_X, val_y) = build_dataset_from_memory(
            memory,
            window_size=self.window_size,
            predict_delta=True  # Ativa a lógica de V(s,a) = delta_r
        )

        if len(train_X) < self.batch_size:
            return None

        train_ds = ValueDataset(train_X, train_y)
        val_ds = ValueDataset(val_X, val_y)

        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False)

        last_train_loss = 0
        last_val_loss = 0

        pbar = tqdm(range(self.epochs), leave=False)

        for epoch in pbar:
            last_train_loss = self.trainer.train_epoch(train_loader)
            last_val_loss = self.trainer.evaluate(val_loader)

            # Atualizamos a barra com os valores atuais de perda
            pbar.set_postfix({"Train Loss": f"{last_train_loss:.5f}", "Val Loss": f"{last_val_loss:.5f}"})

            # Critério de parada antecipada
            if last_train_loss < self.loss_threshold:
                # Opcional: atualizar descrição antes de sair
                pbar.set_description(f"[Value Train] Threshold hit at epoch {epoch}")
                break

        return last_train_loss, last_val_loss

    def predict(self, trajectory):
        """
            trajectory: lista de transições (s, a, r, s_next)
            Onde 'a' é o índice da ação.
            """
        self.model.eval()

        # Verifica se temos dados suficientes para a janela
        if len(trajectory) < self.window_size:
            return None

        window_steps = trajectory[-self.window_size:]

        combined_window = []
        for (s, a, r, s_next) in window_steps:
            action_idx = np.array([a])
            combined_step = np.concatenate([s, action_idx])
            combined_window.append(combined_step)

        # Converte para Tensor (Batch=1, Window, Features)
        window_array = np.array(combined_window)
        X = torch.FloatTensor(window_array).unsqueeze(0).to(self.device)

        with torch.no_grad():
            # A rede prevê o delta_r esperado
            pred_delta = self.model(X)

        return pred_delta.item()
