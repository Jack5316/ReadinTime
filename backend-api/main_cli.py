import argparse
import json
import os
import sys
from pathlib import Path


def load_tts_from_config(device: str, config_path: Path):
    # Ensure local chatterbox/src is importable when running from repo
    repo_root = Path(__file__).parent
    ch_src = repo_root / "chatterbox" / "src"
    if str(ch_src) not in sys.path:
        sys.path.insert(0, str(ch_src))
    try:
        # Correct path when chatterbox/src is on sys.path
        from chatterbox.tts import ChatterboxTTS
    except Exception:
        # Fallback to package-style path if environment differs
        from chatterbox.chatterbox.tts import ChatterboxTTS  # type: ignore

    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        device = (cfg.get("device") or device or "cpu").lower()
        source = (cfg.get("source") or "bundled").lower()

        if source == "bundled":
            bundled_dir = cfg.get("bundled_dir") or "chatterbox_models"
            # Verify required files exist; if not, fall back to HF
            required = ["ve.safetensors", "t3_cfg.safetensors", "s3gen.safetensors", "tokenizer.json"]
            have_all = all((Path(bundled_dir) / f).exists() for f in required)
            if have_all:
                return ChatterboxTTS.from_local(bundled_dir, device)
            # fallback
            print(f"Bundled models missing in {bundled_dir}; falling back to HuggingFace repo")
            return ChatterboxTTS.from_pretrained(device)
        else:
            return ChatterboxTTS.from_pretrained(device)

    # Default
    return ChatterboxTTS.from_pretrained(device)


def main():
    parser = argparse.ArgumentParser(description="Offline Chatterbox TTS CLI")
    parser.add_argument("--text", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--prompt", required=False)
    parser.add_argument("--exaggeration", type=float, default=0.5)
    parser.add_argument("--cfg-weight", type=float, default=0.5)
    parser.add_argument(
        "--device", default=os.getenv("CHB_TTS_DEVICE", "cpu"), help="cpu|cuda|mps"
    )
    parser.add_argument(
        "--config",
        default=str(Path(__file__).with_name("model-config.json")),
        help="Path to model-config.json",
    )
    args = parser.parse_args()

    # Lazy imports to speed startup a bit
    import torch
    import torchaudio

    model = load_tts_from_config(args.device, Path(args.config))

    wav = model.generate(
        args.text,
        audio_prompt_path=args.prompt,
        exaggeration=args.exaggeration,
        cfg_weight=args.cfg_weight,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(out_path), wav, model.sr)
    # Print absolute path for callers
    print(str(out_path.resolve()))
    return 0


if __name__ == "__main__":
    sys.exit(main())


