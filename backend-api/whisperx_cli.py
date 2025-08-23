import argparse
import json
import os
import sys
from pathlib import Path


def group_sentences(words, min_words: int = 8):
    segments = []
    current = []
    count = 0
    for w in words:
        txt = (w.get("text") or w.get("word") or "").strip()
        if not txt:
            continue
        start = float(w.get("start", 0))
        end = float(w.get("end", 0))
        current.append({"text": txt, "start": start, "end": end})
        count += 1
        if (txt.endswith(('.', '!', '?')) and count >= min_words) or count >= min_words * 2:
            seg_text = " ".join([c["text"] for c in current]).strip()
            segments.append({
                "text": seg_text,
                "start": current[0]["start"],
                "end": current[-1]["end"],
            })
            current = []
            count = 0
    if current:
        seg_text = " ".join([c["text"] for c in current]).strip()
        segments.append({
            "text": seg_text,
            "start": current[0]["start"],
            "end": current[-1]["end"],
        })
    return segments


def _resolve_base_dir() -> Path:
    # Prefer the directory of this running module; for Nuitka onefile this is
    # the temporary extraction directory that also holds bundled data files.
    try:
        here = Path(__file__).resolve().parent
        if here.exists():
            return here
    except Exception:
        pass

    # Environment hints provided by onefile bootstrap
    temp_dir = os.environ.get("NUITKA_ONEFILE_TEMP_DIR")
    if temp_dir:
        p = Path(temp_dir)
        if p.exists():
            return p

    parent_dir = os.environ.get("NUITKA_ONEFILE_PARENT")
    if parent_dir:
        p = Path(parent_dir)
        if p.exists():
            return p

    # Frozen executable parent (standalone, non-onefile)
    if getattr(sys, "frozen", False):
        try:
            return Path(sys.argv[0]).resolve().parent
        except Exception:
            return Path(sys.argv[0]).parent

    # Fallback to source file directory
    return Path(__file__).parent


def _resolve_data_dir(base_dir: Path) -> Path:
    # Prefer explicit env vars
    for key in ("ARBOOKS_DATA_DIR", "ARBOOKS_DATA_PATH", "DATA_DIR"):
        val = os.environ.get(key)
        if val:
            p = Path(val)
            if p.exists():
                return p
    # Try common locations relative to executable
    try:
        exe_dir = Path(sys.argv[0]).resolve().parent
        candidates = [
            exe_dir.parent.parent / "Data",  # arBooks/Data when exe at backend-api/dist-cli/
            exe_dir.parent / "Data",        # fallback: backend-api/Data
            base_dir / "Data",
        ]
        for c in candidates:
            if c.exists():
                return c
    except Exception:
        pass
    # Try source tree and cwd
    try:
        src_dir = Path(__file__).resolve().parent
        repo_data = src_dir.parent.parent / "Data"
        if repo_data.exists():
            return repo_data
    except Exception:
        pass
    cwd_data = Path.cwd() / "Data"
    if cwd_data.exists():
        return cwd_data
    # Fallback next to base_dir
    return base_dir / "Data"


def _load_models_config(data_dir: Path) -> dict:
    cfg_path = data_dir / "models.json"
    if cfg_path.exists():
        try:
            return json.loads(cfg_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    # defaults
    return {
        "whisperx": {
            "device": "cpu",
            "models_dir": str(data_dir / "whisperx_models"),
            "asr_model": "small",
            "compute_type": "int8",
            "batch_size": 4
        }
    }


def _maybe_add_local_whisperx(base_dir: Path) -> None:
    # If bundled source exists, add its PARENT directory to sys.path so that
    # `import whisperx` resolves package modules like `whisperx.audio`.
    try:
        # Case 1: Onefile build with data dir mapped as base_dir/whisperx
        pkg_dir = base_dir / "whisperx"
        if (pkg_dir / "__init__.py").exists():
            parent = base_dir
            if str(parent) not in sys.path:
                sys.path.insert(0, str(parent))
            return

        # Case 2: Bundled source under base_dir/whisperX-3.4.2/whisperx
        src_parent = base_dir / "whisperX-3.4.2"
        if (src_parent / "whisperx" / "__init__.py").exists():
            if str(src_parent) not in sys.path:
                sys.path.insert(0, str(src_parent))
            return

        # Fallback: if directories exist, still add reasonable parents
        for candidate in [src_parent, pkg_dir]:
            if candidate.exists():
                parent = candidate if candidate.name != "whisperx" else candidate.parent
                if str(parent) not in sys.path:
                    sys.path.insert(0, str(parent))
                return
    except Exception:
        # Non-fatal; import failure will be handled by caller
        pass


def main():
    parser = argparse.ArgumentParser(description="Offline WhisperX transcription CLI")
    parser.add_argument("--audio", required=False, help="Path to input audio file (wav)")
    parser.add_argument("--outdir", required=False, help="Output directory")
    parser.add_argument("--language", default="auto")
    parser.add_argument("--print-config", action="store_true", help="Print resolved config and exit")
    args = parser.parse_args()

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    base_dir = _resolve_base_dir()
    data_dir = _resolve_data_dir(base_dir)
    cfg = _load_models_config(data_dir)
    wx_cfg = cfg.get("whisperx", {})

    # Prefer models directory from config, else Data/whisperx_models
    raw_models_dir = wx_cfg.get("models_dir", "whisperx_models")
    models_path = Path(raw_models_dir)
    models_dir = models_path if models_path.is_absolute() else (data_dir / models_path)
    if models_dir.exists():
        os.environ.setdefault("HF_HOME", str(models_dir))
        os.environ.setdefault("TRANSFORMERS_CACHE", str(models_dir))
        os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(models_dir))
        os.environ.setdefault("CT2_CACHES_DIR", str(models_dir))
        download_root = str(models_dir)
    else:
        download_root = None

    # Dev utility: print resolved config without importing whisperx or requiring audio
    if args.print_config:
        summary = {
            "base_dir": str(base_dir),
            "data_dir": str(data_dir),
            "whisperx_config": wx_cfg,
            "models_dir": str(models_dir),
            "download_root": download_root,
        }
        print(json.dumps(summary, indent=2))
        return 0

    if not args.audio or not args.outdir:
        print("--audio and --outdir are required unless --print-config is used", file=sys.stderr)
        return 2

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Allow importing bundled whisperx source if provided
    _maybe_add_local_whisperx(base_dir)

    try:
        import whisperx  # type: ignore
    except Exception as e:
        print(f"ERROR: whisperx not available: {e}")
        return 1

    # Load and transcribe
    audio = whisperx.load_audio(args.audio)

    asr_model = wx_cfg.get("asr_model", "small")
    device = wx_cfg.get("device", "cpu")
    compute_type = wx_cfg.get("compute_type", "int8")
    batch_size = int(wx_cfg.get("batch_size", 4))

    if download_root:
        model = whisperx.load_model(asr_model, device=device, compute_type=compute_type, download_root=download_root)
    else:
        model = whisperx.load_model(asr_model, device=device, compute_type=compute_type)

    transcribe_language = None if args.language == "auto" else args.language
    if transcribe_language:
        result = model.transcribe(audio, batch_size=batch_size, language=transcribe_language)  # type: ignore
    else:
        result = model.transcribe(audio, batch_size=batch_size)  # type: ignore

    # Optional alignment
    language = result.get("language")
    segments_for_align = result.get("segments") or []
    if (
        bool(wx_cfg.get("align", True))
        and segments_for_align
        and language in ["en", "fr", "de", "es", "it", "ja", "zh", "nl", "uk", "pt"]
    ):
        # Alignment API expects model_dir, not download_root
        model_dir_arg = {"model_dir": str(models_dir)} if models_dir.exists() else {}
        model_a, metadata = whisperx.load_align_model(language_code=language, device=device, **model_dir_arg)
        result = whisperx.align(segments_for_align, model_a, metadata, audio, device=device, return_char_alignments=False)

    segments = result.get("segments", [])
    words = []
    for seg in segments:
        for w in seg.get("words", []):
            words.append({
                "text": w.get("word", ""),
                "start": float(w.get("start", 0)),
                "end": float(w.get("end", 0)),
                "confidence": float(w.get("score", 0)),
            })

    mappings = group_sentences(words)
    with open(outdir / "text_mappings.json", "w", encoding="utf-8") as f:
        json.dump(mappings, f, indent=2)

    with open(outdir / "transcription.json", "w", encoding="utf-8") as f:
        json.dump({"language": language, "segments": segments, "words": words}, f, indent=2)

    print(str((outdir / "text_mappings.json").resolve()))
    return 0


if __name__ == "__main__":
    sys.exit(main())


