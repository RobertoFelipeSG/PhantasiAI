import os
import pymupdf4llm
from pathlib import Path

DOCS_DIR = Path(__file__).resolve().parent.parent/"tests"/"documents"

def convert_pdf_to_txt(pdf_path: Path, txt_path: Path):
    try:
        # Convert PDF → Markdown text
        md_text = pymupdf4llm.to_markdown(str(pdf_path))
        # Save as UTF-8 plain text file
        txt_path.write_text(md_text, encoding="utf-8")

        print(f"Converted {pdf_path}")
    except Exception as e:
        print(f"Could not convert {pdf_path}: {e}")

def main():
    for pdf_path in DOCS_DIR.rglob("*pdf"):
        txt_path = pdf_path.with_suffix(".txt")
        if not txt_path.exists():
            convert_pdf_to_txt(pdf_path, txt_path)

if __name__ == "__main__":
    main()