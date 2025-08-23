# Services package
# Keep this package lightweight to avoid importing heavy optional dependencies
# during tools like packagers/compilers. Import submodules explicitly where
# needed, e.g. `from services.pdf_processor import PDFProcessor`.

__all__: list[str] = []