from dataclasses import dataclass, fields
from typing import Any


@dataclass
class TrainConfig:
    pca_components: int = 32
    logistic_lr: float = 0.5
    logistic_epochs: int = 400
    logistic_reg: float = 1e-3
    linear_reg_reg: float = 1e-3
    train_ratio: float = 0.8
    random_seed: int = 42
    visualize_samples: int = 5
    vector_length: int = 360
    checkpoint_dir: str = "checkpoints"
    similarity_threshold: float = 0.5
    odom_columns: tuple[str, ...] = ("x_car", "y_car", "heading_car", "x_linear_vel", "y_linear_vel", "w_angular_vel")


@dataclass
class TestConfig:
    threshold: float = 0.5
    vector_length: int = 360
    random_seed: int = 42
    checkpoint_dir: str = "checkpoints"
    odom_columns: tuple[str, ...] = ("x_car", "y_car", "heading_car", "x_linear_vel", "y_linear_vel", "w_angular_vel")


def update_config_from_args(config: Any, args: Any) -> None:
    for field_def in fields(config):
        if hasattr(args, field_def.name) and getattr(args, field_def.name) is not None:
            setattr(config, field_def.name, getattr(args, field_def.name))


def get_default_train_config() -> TrainConfig:
    return TrainConfig()


def get_default_test_config() -> TestConfig:
    return TestConfig()
