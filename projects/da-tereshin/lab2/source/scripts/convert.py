import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import pymupdf
import pymupdf4llm
import yaml


def extract_pages(pdf: Path, ocr: bool) -> list[dict]:
    if not ocr:
        return pymupdf4llm.to_markdown(str(pdf), page_chunks=True, use_ocr=False)

    tessdata = Path(sys.prefix) / "share/tessdata"
    with pymupdf.open(pdf) as original, pymupdf.open() as recognized:
        for number, page in enumerate(original, start=1):
            print(f"OCR: {number}/{len(original)}", flush=True)
            image = page.get_pixmap(dpi=300, colorspace=pymupdf.csRGB, alpha=False)
            ocr_pdf = image.pdfocr_tobytes(language="rus+eng", tessdata=str(tessdata))
            with pymupdf.open(stream=ocr_pdf, filetype="pdf") as single_page:
                recognized.insert_pdf(single_page)
        return pymupdf4llm.to_markdown(recognized, page_chunks=True, use_ocr=False)


def convert(pdf: Path, output: Path, ocr: bool) -> None:
    pages = extract_pages(pdf, ocr)
    target = output / pdf.stem
    target.mkdir(parents=True, exist_ok=True)
    # При повторном запуске заменяем страницы этого документа
    for old_page in target.glob("*.md"):
        if old_page.is_file() and old_page.stem.isdecimal():
            old_page.unlink()

    converted_at = datetime.now(timezone.utc).isoformat()
    for number, page in enumerate(pages, start=1):
        text = page["text"].strip()
        metadata = {
            "source": pdf.name,
            "source_path": pdf.resolve().as_posix(),
            "page": number,
            "page_count": len(pages),
            "converted_at": converted_at,
            "extraction_method": "ocr" if ocr else "text_layer",
        }
        if ocr:
            metadata.update(ocr_engine="tesseract", ocr_language="rus+eng", ocr_dpi=300)
        header = yaml.safe_dump(metadata, allow_unicode=True, sort_keys=False).strip()
        (target / f"{number:04d}.md").write_text(
            f"---\n{header}\n---\n\n{text}\n", encoding="utf-8"
        )
        if not text:
            print(f"Предупреждение: {pdf.name}, страница {number}: нет текста")
    print(f"{pdf.name}: {len(pages)} страниц → {target}")


def convert_corpus(config_path: Path) -> None:
    root = config_path.resolve().parent
    options = yaml.safe_load(config_path.read_text(encoding="utf-8"))["conversion"]
    output = root / options["output_dir"]
    documents = options["documents"]
    for number, document in enumerate(documents, start=1):
        pdf = root / document["path"]
        print(f"Документ {number}/{len(documents)}: {pdf.name}", flush=True)
        convert(pdf, output, ocr=document["ocr"])
    print(f"Готово: обработано документов — {len(documents)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PDF → Markdown по конфигу")
    parser.add_argument(
        "--config", type=Path,
        default=Path(__file__).resolve().parents[2] / "config.yaml",
    )
    convert_corpus(parser.parse_args().config)