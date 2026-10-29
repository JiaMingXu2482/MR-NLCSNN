"""Inference-only loader. No training runner, data loader or CSV dependency."""
import hashlib,sys
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parent/'frozen_model'))
from model_capacity_frozen import CapacityModel

def build(arm):
    if arm!='A': raise ValueError('This audit freezes A only')
    return CapacityModel(gru_hidden=16,mlp_hidden=48,history_di=True)

base=SimpleNamespace(digest=lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest())
