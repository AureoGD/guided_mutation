import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------
# Activation factory
# ----------------------------------------
def get_activation(name: str):
    name = name.lower()

    if name == "relu":
        return nn.ReLU()
    elif name == "tanh":
        return nn.Tanh()
    elif name == "elu":
        return nn.ELU()
    elif name == "leaky_relu":
        return nn.LeakyReLU()
    elif name == "sigmoid":
        return nn.Sigmoid()
    else:
        raise ValueError(f"Unsupported activation: {name}")


# ----------------------------------------
# Policy
# ----------------------------------------
class Policy(nn.Module):

    def __init__(self, env_spec, model_cfg):
        super().__init__()

        # -----------------------------
        # Validate config
        # -----------------------------
        if "layers" not in model_cfg:
            raise ValueError("model_cfg must contain 'layers'")

        # -----------------------------
        # Env info
        # -----------------------------
        self.obs_dim = env_spec.obs_dim
        self.act_dim = env_spec.act_dim
        self.is_discrete = env_spec.is_discrete

        # -----------------------------
        # Build network
        # -----------------------------
        layers_cfg = model_cfg["layers"]

        layers = []
        in_dim = self.obs_dim

        for layer_cfg in layers_cfg:
            out_dim = layer_cfg["units"]
            activation = get_activation(layer_cfg["activation"])

            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(activation)

            in_dim = out_dim

        # Output layer
        layers.append(nn.Linear(in_dim, self.act_dim))

        self.net = nn.Sequential(*layers)

        # -----------------------------
        # Weight initialization
        # -----------------------------
        self.apply(self._init_weights)

        # -----------------------------
        # Continuous action scaling
        # -----------------------------
        self._low = None
        self._high = None

        if not self.is_discrete:
            if "action_low" in model_cfg and "action_high" in model_cfg:
                low = np.asarray(model_cfg["action_low"], dtype=np.float32).ravel()
                high = np.asarray(model_cfg["action_high"], dtype=np.float32).ravel()

                if low.shape != high.shape or low.size != self.act_dim:
                    raise ValueError("action_low/high must match action dimension")

                self._low = torch.tensor(low, dtype=torch.float32)
                self._high = torch.tensor(high, dtype=torch.float32)

    # ----------------------------------------
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight)
            nn.init.zeros_(module.bias)

    # ----------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 1:
            x = x.unsqueeze(0)
        return self.net(x)

    # ----------------------------------------
    @torch.no_grad()
    def predict(self, state, device="cpu", deterministic=True):

        # -----------------------------
        # Convert input
        # -----------------------------
        if isinstance(state, np.ndarray):
            x = torch.from_numpy(state).float()
        else:
            x = state.float()

        single = False
        if x.ndim == 1:
            x = x.unsqueeze(0)
            single = True

        x = x.to(device)

        # -----------------------------
        # Forward
        # -----------------------------
        out = self.forward(x)

        # -----------------------------
        # Discrete
        # -----------------------------
        if self.is_discrete:
            probs = F.softmax(out, dim=-1)

            if deterministic:
                action = probs.argmax(dim=-1)
            else:
                action = torch.multinomial(probs, 1).squeeze(-1)

            action = action.cpu().numpy()
            return int(action[0]) if single else action, state

        # -----------------------------
        # Continuous
        # -----------------------------
        action = torch.tanh(out)

        if self._low is not None and self._high is not None:
            low = self._low.to(action.device)
            high = self._high.to(action.device)
            action = low + (action + 1.0) * 0.5 * (high - low)

        action = action.cpu().numpy().astype(np.float32)

        return action[0] if single else action, state

    # ----------------------------------------
    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())

    def set_parameters(self, flat_params):

        idx = 0
        flat_params = torch.tensor(flat_params, dtype=torch.float32)

        for param in self.parameters():
            numel = param.numel()

            param.data.copy_(flat_params[idx:idx + numel].view_as(param))

            idx += numel

    def get_parameters(self):

        return torch.cat([p.data.view(-1) for p in self.parameters()]).cpu().numpy()
