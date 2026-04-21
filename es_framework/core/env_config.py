from dataclasses import dataclass
from typing import Type, Dict, Any


@dataclass
class EnvConfig:
    # ------------------------
    # Classes principais
    # ------------------------
    env_class: Type
    controller_class: Type
    task_class: Type

    # ------------------------
    # Caminhos / arquivos
    # ------------------------
    scene_path: str
    urdf_path: str
    tpe_model_path: str

    # ------------------------
    # Parâmetros
    # ------------------------
    normalizer_params: Dict[str, Any]

    # ------------------------
    # Simulação
    # ------------------------
    render: bool = False
