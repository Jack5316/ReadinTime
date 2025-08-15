import argparse
import json
import os
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Offline PDF -> Markdown CLI using project pdf_processor")
    parser.add_argument("--pdf", required=True, help="Input PDF path")
    parser.add_argument("--outdir", required=True, help="Output directory where pdf_result.md will be written")
    args = parser.parse_args()

    project_root = Path(__file__).parent
    sys.path.insert(0, str(project_root))

    from services.pdf_processor import PDFProcessor

    pdf_path = args.pdf
    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)

    proc = PDFProcessor()
    result = proc._run_pdf_command_sync(
        "",  # unused, we call the direct methods below when possible
        pdf_path,
        timeout=120,
    )
    # We won't use the above output; just call the high-level convert function directly
    res = proc.convert_pdf_to_markdown(pdf_path, outdir)
    # convert_pdf_to_markdown may be async; handle awaitable result
    if hasattr(res, "__await__"):
        import asyncio
        res = asyncio.get_event_loop().run_until_complete(res)

    print(json.dumps(res))
    return 0 if res and res.get("success") else 1


if __name__ == "__main__":
    sys.exit(main())


