"""
Lightweight text extractor for PDF, DOCX and Excel.
"""
from pathlib import Path
import pdfplumber, docx2txt, pandas as pd

def extract_text(file_path: str) -> str:
    path = Path(file_path)
    ext = path.suffix.lower()

    if ext == ".pdf":
        with pdfplumber.open(path) as pdf:
            return "\n".join((p.extract_text() or "") for p in pdf.pages)

    if ext in {".docx", ".doc"}:
        return docx2txt.process(path)

    if ext in {".xls", ".xlsx"}:
        dfs = pd.read_excel(path, sheet_name=None)
        return "\n".join(df.to_csv(index=False) for df in dfs.values())

    # fall back to plain text
    return path.read_text(encoding="utf-8", errors="ignore")
