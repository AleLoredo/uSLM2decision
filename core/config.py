"""
OpenJEV Configuration Management Module
Implements centralized, validated configuration via config.yaml using Pydantic.
"""

import os
from typing import Literal, Optional, List
from pathlib import Path
import yaml
from pydantic import BaseModel, Field

DEFAULT_CONFIG_PATH = "config.yaml"

class ModelConfig(BaseModel):
    base_dir: str = "./models"
    active_model: str = "qwen2.5-0.5b"
    backbone_path: str = "./models/qwen2.5-0.5b"
    heads_path: str = "./models/qwen2.5-0.5b/heads.safetensors"
    calibration_path: Optional[str] = "./models/qwen2.5-0.5b/calibration.json"
    backend: Literal["mlx", "torch"] = "mlx"
    device: str = "auto"

    def resolve_paths(self) -> "ModelConfig":
        """Resolves relative paths if configured relative to base_dir or active_model."""
        if not self.backbone_path or self.backbone_path == "./models/qwen2.5-0.5b":
            self.backbone_path = os.path.join(self.base_dir, self.active_model)
        if not self.heads_path or self.heads_path == "./models/qwen2.5-0.5b/heads.safetensors":
            self.heads_path = os.path.join(self.backbone_path, "heads.safetensors")
        return self

class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000

class OpenJEVConfig(BaseModel):
    mode: Literal["mock", "real"] = "mock"
    model: ModelConfig = Field(default_factory=ModelConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)

    def validate_real_paths(self) -> List[str]:
        """Returns a list of missing paths if mode == 'real'."""
        missing = []
        if self.mode == "real":
            if not os.path.exists(self.model.backbone_path):
                missing.append(f"backbone_path not found: {self.model.backbone_path}")
            if not os.path.exists(self.model.heads_path):
                missing.append(f"heads_path not found: {self.model.heads_path}")
        return missing

def load_config(config_path: str = DEFAULT_CONFIG_PATH) -> OpenJEVConfig:
    """Loads configuration from YAML file or returns default OpenJEVConfig."""
    path = Path(config_path)
    if path.is_file():
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        config = OpenJEVConfig(**data)
    else:
        config = OpenJEVConfig()
    return config

def save_config(config: OpenJEVConfig, config_path: str = DEFAULT_CONFIG_PATH) -> None:
    """Serializes OpenJEVConfig to YAML file."""
    path = Path(config_path)
    data = config.model_dump()
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="OpenJEV Configuration Tool")
    parser.add_argument("--set-mode", choices=["mock", "real"], help="Set execution mode")
    parser.add_argument("--set-model", help="Set active model name")
    parser.add_argument("--set-backbone", help="Set backbone path")
    parser.add_argument("--set-heads", help="Set heads path")
    parser.add_argument("--config-file", default=DEFAULT_CONFIG_PATH)

    args = parser.parse_args()
    cfg = load_config(args.config_file)
    if args.set_mode:
        cfg.mode = args.set_mode
    if args.set_model:
        cfg.model.active_model = args.set_model
        cfg.model.resolve_paths()
    if args.set_backbone:
        cfg.model.backbone_path = args.set_backbone
    if args.set_heads:
        cfg.model.heads_path = args.set_heads

    save_config(cfg, args.config_file)
    print(f"[OpenJEV Config] Saved configuration to {args.config_file} (mode='{cfg.mode}', model='{cfg.model.active_model}')")
