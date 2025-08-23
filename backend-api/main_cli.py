import argparse
import json
import os
import sys
from pathlib import Path


def _resolve_under(base: Path, val: str | None) -> str | None:
    if not val:
        return None
    p = Path(val)
    return str(p if p.is_absolute() else (base / p))


def _find_data_dir() -> Path | None:
    """Locate the shared Data directory.

    Priority order:
    1) ARBOOKS_DATA_DIR env var (or ARBOOKS_DATA_PATH)
    2) Relative to executable (dist-cli/ -> backend-api/ -> repo root / Data)
    3) Relative to this file when running from source (repo root / Data)
    4) Current working directory / Data
    """
    env_dir = os.environ.get("ARBOOKS_DATA_DIR") or os.environ.get("ARBOOKS_DATA_PATH")
    if env_dir:
        p = Path(env_dir)
        if p.exists():
            return p
    # Executable location (Nuitka onefile places the exe in dist dir)
    try:
        exe_dir = Path(sys.argv[0]).resolve().parent
        candidates = [
            exe_dir / "Data",               # Data next to the .exe (preferred for distribution)
            exe_dir.parent / "Data",        # backend-api/Data when exe at backend-api/dist-cli/
            exe_dir.parent.parent / "Data",  # arBooks/Data when exe at backend-api/dist-cli/
        ]
        for c in candidates:
            if c.exists():
                return c
    except Exception:
        pass
    # Source tree relative
    try:
        src_dir = Path(__file__).resolve().parent
        repo_data = src_dir.parent / "Data"
        if repo_data.exists():
            return repo_data
    except Exception:
        pass
    # CWD/Data
    cwd_data = Path.cwd() / "Data"
    if cwd_data.exists():
        return cwd_data
    return None


def _resolve_bundled_dir(bundled_dir_value: str | None, config_path: Path, data_dir: Path | None) -> Path:
    """Resolve bundled_dir to an absolute path, preferring Data/ when provided."""
    candidate = bundled_dir_value or "chatterbox_models"
    cand_path = Path(candidate)
    if cand_path.is_absolute():
        return cand_path
    if data_dir is not None and (data_dir / cand_path).exists():
        return (data_dir / cand_path).resolve()
    if (config_path.parent / cand_path).exists():
        return (config_path.parent / cand_path).resolve()
    # Fallback to Data if given (even if not existing yet)
    if data_dir is not None:
        return (data_dir / cand_path).resolve()
    return (config_path.parent / cand_path).resolve()


def _load_effective_config(explicit_config_path: Path | None) -> tuple[dict, Path | None, Path | None]:
    """Load model configuration by merging Data/models.json (section 'chatterbox') over model-config.json.

    Returns: (config_dict, resolved_config_path, data_dir)
    """
    data_dir = _find_data_dir()
    # Determine config path
    resolved_cfg: Path | None = None
    if explicit_config_path and explicit_config_path.exists():
        resolved_cfg = explicit_config_path
    else:
        # Prefer Data/model-config.json, then sibling model-config.json
        if data_dir and (data_dir / "model-config.json").exists():
            resolved_cfg = data_dir / "model-config.json"
        else:
            sibling = Path(__file__).with_name("model-config.json")
            if sibling.exists():
                resolved_cfg = sibling

    base_cfg: dict = {}
    if resolved_cfg and resolved_cfg.exists():
        with open(resolved_cfg, "r", encoding="utf-8") as f:
            try:
                base_cfg = json.load(f) or {}
            except Exception:
                base_cfg = {}

    # Optional overlay from Data/models.json -> section 'chatterbox'
    if data_dir and (data_dir / "models.json").exists():
        try:
            with open(data_dir / "models.json", "r", encoding="utf-8") as f:
                models_cfg = json.load(f) or {}
            ch_cfg = models_cfg.get("chatterbox") or {}
            # Only set keys not already present to respect explicit config
            for k in ("device", "source", "bundled_dir", "hf_repo", "target_sr"):
                if k not in base_cfg and k in ch_cfg:
                    base_cfg[k] = ch_cfg[k]
        except Exception:
            pass
    # Normalize any relative paths under Data/
    if data_dir:
        for key in ("bundled_dir", "ov_dir", "models_dir"):
            if key in base_cfg and isinstance(base_cfg[key], str):
                base_cfg[key] = _resolve_under(data_dir, base_cfg[key])

    return base_cfg, resolved_cfg, data_dir


class OpenVinoChatterboxTTS:
    """Placeholder OpenVINO-backed TTS wrapper.

    For now, this class only validates presence of OpenVINO IR files and holds device
    settings. The actual inference path can be implemented incrementally (e.g., split
    into voice encoder / t3 / s3gen subgraphs) while keeping this interface stable.
    """

    def __init__(self, ov_dir: Path, device: str = "AUTO:NPU,CPU", sr: int = 24000):
        from openvino.runtime import Core  # type: ignore
        self.core = Core()
        self.device = device
        self.ov_dir = ov_dir
        self.sr = sr

        # Detect any of the expected IRs to validate setup. Actual loading of
        # components will depend on the final split we export.
        expected_any = [
            ov_dir / "model.xml",
            ov_dir / "t3.xml",
            ov_dir / "s3gen.xml",
            ov_dir / "voice_encoder.xml",
        ]
        if not any(p.exists() for p in expected_any):
            raise FileNotFoundError(
                f"No OpenVINO IRs found in {ov_dir}. Expected one of: model.xml/t3.xml/s3gen.xml/voice_encoder.xml"
            )

        # NOTE: We don't compile models here yet to avoid hard-coding interfaces.
        # Future work: compile subgraphs and wire tokenization/audio prompt processing.

    @classmethod
    def from_ir(cls, ov_dir: str | Path, device: str = "AUTO:NPU,CPU", sr: int = 24000) -> "OpenVinoChatterboxTTS":
        return cls(Path(ov_dir).resolve(), device=device, sr=sr)

    def generate(self, *args, **kwargs):  # signature mirrors ChatterboxTTS.generate
        # Placeholder: Until the OpenVINO graphs are wired, fail clearly.
        raise NotImplementedError(
            "OpenVINO backend is detected but not yet implemented. Export IRs and wire subgraphs, "
            "or switch backend to 'pytorch' to use the existing implementation."
        )


def load_tts_from_config(device: str, config_path: Path):
    # Prefer Data/chatterbox/src when present; when compiled, avoid repo chatterbox/src to ensure Data takes precedence
    try:
        data_dir_for_imports = _find_data_dir()
        if data_dir_for_imports is not None:
            data_ch_src = (data_dir_for_imports / "chatterbox" / "src").resolve()
            if data_ch_src.exists() and str(data_ch_src) not in sys.path:
                # Insert at front so Data version is preferred
                sys.path.insert(0, str(data_ch_src))
    except Exception:
        # Best-effort, non-fatal if Data is missing
        pass

    # Import from the canonical module path. Avoid fallback paths that confuse
    # static analysis tools like Nuitka during import following.
    try:
        from chatterbox.tts import ChatterboxTTS
    except ImportError as e:
        print(f"Failed to import ChatterboxTTS: {e}")
        print("Please ensure PyTorch and chatterbox dependencies are installed:")
        print("pip install torch torchaudio transformers safetensors")
        raise SystemExit(1)

    # Merge Data/models.json over config, and resolve bundled_dir to Data when present
    cfg, resolved_cfg_path, data_dir = _load_effective_config(config_path)

    # Optional: choose backend (default pytorch). If OpenVINO requested and IRs exist, use OV wrapper; else fall back.
    backend = (cfg.get("backend") or os.getenv("CHB_BACKEND") or "pytorch").lower()
    ov_device = cfg.get("ov_device") or os.getenv("CHB_OV_DEVICE") or "AUTO:NPU,CPU"
    ov_dir = cfg.get("openvino_dir") or "chatterbox_openvino"

    # Ensure HF caches are directed to Data/chatterbox_models when available
    try:
        if data_dir:
            bundled_dir = cfg.get("bundled_dir") or "chatterbox_models"
            models_dir = (data_dir / bundled_dir).resolve()
            models_dir.mkdir(parents=True, exist_ok=True)
            os.environ.setdefault("HF_HOME", str(models_dir))
            os.environ.setdefault("TRANSFORMERS_CACHE", str(models_dir))
            os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(models_dir))
    except Exception:
        # Best-effort cache redirection
        pass

    # Final device selection (env/arg wins over file if provided)
    device = (device or cfg.get("device") or os.getenv("CHB_TTS_DEVICE") or "cpu").lower()
    source = (cfg.get("source") or "bundled").lower()

    # Try to initialize OpenVINO backend if requested
    if backend == "openvino":
        try:
            if data_dir is not None:
                ov_root = (data_dir / ov_dir).resolve()
            else:
                ov_root = Path.cwd() / ov_dir
            if ov_root.exists():
                try:
                    # If OV IRs exist, build OV wrapper; may raise NotImplementedError at generate()
                    return OpenVinoChatterboxTTS.from_ir(ov_root, device=ov_device, sr=int(cfg.get("target_sr", 24000)))
                except FileNotFoundError as e:
                    print(f"OpenVINO requested but IRs missing: {e}; falling back to PyTorch backend.")
            else:
                print(f"OpenVINO directory {ov_root} not found; falling back to PyTorch backend.")
        except Exception as e:
            print(f"OpenVINO backend init failed ({e}); falling back to PyTorch backend.")

    if source == "bundled":
        bundled_dir_resolved = _resolve_bundled_dir(cfg.get("bundled_dir"), resolved_cfg_path or Path.cwd(), data_dir)
        # Verify required files exist; if not, fall back to HF
        required = ["ve.safetensors", "t3_cfg.safetensors", "s3gen.safetensors", "tokenizer.json"]
        have_all = all((bundled_dir_resolved / f).exists() for f in required)
        if have_all:
            try:
                print(f"Loading bundled models from: {bundled_dir_resolved}")
                return ChatterboxTTS.from_local(str(bundled_dir_resolved), device)
            except Exception as e:
                print(f"Failed to load bundled models: {e}")
                print("Falling back to HuggingFace repo...")
                return ChatterboxTTS.from_pretrained(device)
        print(f"Bundled models missing in {bundled_dir_resolved}; falling back to HuggingFace repo")
        return ChatterboxTTS.from_pretrained(device)

    return ChatterboxTTS.from_pretrained(device)


def main():
    parser = argparse.ArgumentParser(description="Offline Chatterbox TTS CLI")
    parser.add_argument("--text", required=False)
    parser.add_argument("--text-file", required=False, help="Path to a UTF-8 text file to read input from")
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
        help=(
            "Path to model-config.json. If omitted or missing, the CLI will look for Data/model-config.json "
            "or merge settings from Data/models.json (section 'chatterbox') when available."
        ),
    )
    args = parser.parse_args()

    # Lazy imports to speed startup a bit
    import torch
    HAS_TORCHAUDIO = True
    try:
        import torchaudio  # type: ignore
    except Exception:
        HAS_TORCHAUDIO = False

    model = load_tts_from_config(args.device, Path(args.config))

    # Load text from file if provided to avoid command-line length limits
    input_text = args.text
    if not input_text and args.text_file:
        with open(args.text_file, "r", encoding="utf-8") as f:
            input_text = f.read()
    if not input_text:
        raise SystemExit("No input text provided. Use --text or --text-file.")

    wav = model.generate(
        input_text,
        audio_prompt_path=args.prompt,
        exaggeration=args.exaggeration,
        cfg_weight=args.cfg_weight,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if HAS_TORCHAUDIO:
        # Save using torchaudio when available
        import torchaudio  # type: ignore
        torchaudio.save(str(out_path), wav, model.sr)
    else:
        # Fallback: save using wave + numpy (mono 16-bit)
        import numpy as np
        import wave
        wav_np = wav.squeeze(0).detach().cpu().numpy()
        wav_np = np.clip(wav_np, -1.0, 1.0)
        wav_int16 = (wav_np * 32767.0).astype(np.int16)
        with wave.open(str(out_path), 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(int(model.sr))
            wf.writeframes(wav_int16.tobytes())
    # Print absolute path for callers
    print(str(out_path.resolve()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
