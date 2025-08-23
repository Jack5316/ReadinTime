# export_chatterbox_ov.py
import os
from pathlib import Path
import torch
import openvino as ov
from openvino.runtime import serialize

# Point to Data
repo_root = Path(__file__).resolve().parent
data_dir = (repo_root.parent / "Data").resolve()
bundled = data_dir / "chatterbox_models/models--ResembleAI--chatterbox/snapshots/1b475dffa71fb191cb6d5901215eb6f55635a9b6"
ov_out = data_dir / "chatterbox_openvino"
ov_out.mkdir(parents=True, exist_ok=True)

# Make chatterbox importable
ch_src = data_dir / "chatterbox" / "src"
import sys
sys.path.insert(0, str(ch_src))

from chatterbox.tts import ChatterboxTTS  # loads PyTorch modules internally

# Ensure HF caches go to Data
os.environ.setdefault("HF_HOME", str(data_dir / "chatterbox_models"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(data_dir / "chatterbox_models"))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(data_dir / "chatterbox_models"))

# 1) Load local model
tts = ChatterboxTTS.from_local(str(bundled), device="cpu")
tts.eval()

# TODO: Extract the actual submodules from tts (these attributes may differ; inspect tts to find them)
# For illustration only:
voice_enc = getattr(tts, "voice_encoder", None)
t3 = getattr(tts, "t3", None)
s3gen = getattr(tts, "s3gen", None)

def to_ov(model, example_inputs, dst_xml):
    model.eval()
    with torch.no_grad():
        ov_model = ov.convert_model(model, example_input=example_inputs)
    serialize(ov_model, str(dst_xml))

# 2) Export voice encoder
if voice_enc is not None:
    # Prepare an example input matching your voice encoder forward signature (e.g., mel [B, T, n_mels])
    import torch
    example = torch.randn(1, 400, 80)  # placeholder; adjust to real shape
    to_ov(voice_enc, example, ov_out / "voice_encoder.xml")
    print("Exported voice_encoder.xml")

# 3) Export T3
if t3 is not None:
    # Example: token ids [B, T] and maybe attention masks; adjust to real expected inputs
    ids = torch.randint(0, 1000, (1, 128))
    to_ov(t3, (ids,), ov_out / "t3.xml")
    print("Exported t3.xml")

# 4) Export S3gen
if s3gen is not None:
    # Example: hidden reps + speaker embedding; replace with real tensors/shapes
    hid = torch.randn(1, 256, 128)
    spk = torch.randn(1, 256)
    to_ov(s3gen, (hid, spk), ov_out / "s3gen.xml")
    print("Exported s3gen.xml")