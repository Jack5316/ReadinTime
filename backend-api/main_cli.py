import argparse
import json
import os
import sys
from pathlib import Path


def load_tts_from_config(device: str, config_path: Path):
    from chatterbox.src.chatterbox.tts import ChatterboxTTS

    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        device = (cfg.get("device") or device or "cpu").lower()
        source = (cfg.get("source") or "bundled").lower()

        if source == "bundled":
            bundled_dir = cfg.get("bundled_dir") or "chatterbox_models"
            return ChatterboxTTS.from_local(bundled_dir, device)
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


