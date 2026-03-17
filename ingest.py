"""
ingest.py — OCR, chunk, and upsert petrophysics PDFs into the Chroma vectorstore.

Usage:
    python ingest.py                         # process all PDFs in ./content/
    python ingest.py path/to/file.pdf        # process a single file
    python ingest.py --force                 # re-ingest already-processed files
    python ingest.py --list                  # show ingested files
    python ingest.py --pages 10 25 file.pdf  # OCR only pages 10–25
"""

import argparse
import json
import sys
from pathlib import Path

import pytesseract
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pdf2image import convert_from_path

# ── Configuration ─────────────────────────────────────────────────────────────

CONTENT_DIR   = Path("./content")
CHROMA_DIR    = Path("./chroma_db")
REGISTRY_FILE = Path("./ingested.json")   # tracks which files were processed

KEYWORDS = [
    "neutron", "density", "porosity", "nphi", "rhob",
    "cross plot", "gamma", "lithology", "permeability",
    "saturation", "resistivity", "formation", "sandstone",
    "limestone", "dolomite", "shale",
]

EMBED_MODEL      = "nomic-embed-text"
COLLECTION_NAME  = "petrophysics_refs"
CHUNK_SIZE       = 800
CHUNK_OVERLAP    = 150
OCR_DPI          = 200


# ── Registry helpers ──────────────────────────────────────────────────────────

def load_registry() -> dict:
    if REGISTRY_FILE.exists():
        return json.loads(REGISTRY_FILE.read_text())
    return {}


def save_registry(registry: dict) -> None:
    REGISTRY_FILE.write_text(json.dumps(registry, indent=2))


# ── OCR ───────────────────────────────────────────────────────────────────────

def ocr_pdf(pdf_path: Path, first_page: int = 1, last_page: int = None) -> str:
    print(f"  OCR: {pdf_path.name} (pages {first_page}–{last_page or 'end'}, {OCR_DPI} dpi)")
    pages = convert_from_path(
        str(pdf_path),
        dpi=OCR_DPI,
        first_page=first_page,
        last_page=last_page,
    )
    texts = []
    for i, page in enumerate(pages, start=first_page):
        text = pytesseract.image_to_string(page, lang="eng")
        texts.append(f"\n\n--- Page {i} ---\n{text}")
        print(f"    page {i} done ({len(text)} chars)")
    return "\n".join(texts)


# ── Chunking ──────────────────────────────────────────────────────────────────

def chunk_and_filter(text: str, source_name: str) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    raw_chunks = splitter.split_text(text)

    docs = []
    for i, chunk in enumerate(raw_chunks):
        if any(k in chunk.lower() for k in KEYWORDS):
            docs.append(Document(
                page_content=chunk,
                metadata={
                    "type": "reference",
                    "source": source_name,
                    "chunk_id": i,
                },
            ))

    print(f"  Chunks: {len(raw_chunks)} total → {len(docs)} after keyword filter")
    return docs


# ── Vectorstore ───────────────────────────────────────────────────────────────

def get_vectorstore() -> Chroma:
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )


def upsert_docs(docs: list[Document]) -> None:
    if not docs:
        print("  No documents to upsert.")
        return
    vs = get_vectorstore()
    vs.add_documents(docs)
    print(f"  Upserted {len(docs)} chunks into '{COLLECTION_NAME}'.")


# ── Main ingest logic ─────────────────────────────────────────────────────────

def ingest_file(
    pdf_path: Path,
    source_label: str = None,
    first_page: int = 1,
    last_page: int = None,
    force: bool = False,
) -> None:
    registry = load_registry()
    key = str(pdf_path.resolve())

    if key in registry and not force:
        print(f"  Skipping (already ingested). Use --force to re-ingest.")
        return

    source_label = source_label or pdf_path.stem
    text = ocr_pdf(pdf_path, first_page=first_page, last_page=last_page)
    docs = chunk_and_filter(text, source_name=source_label)
    upsert_docs(docs)

    registry[key] = {
        "source_label": source_label,
        "chunks_added": len(docs),
        "pages": f"{first_page}–{last_page or 'end'}",
    }
    save_registry(registry)
    print(f"  Done. Registry updated.")


def list_ingested() -> None:
    registry = load_registry()
    if not registry:
        print("No files ingested yet.")
        return
    print(f"{'File':<50} {'Source':<30} {'Chunks':>6}  Pages")
    print("-" * 100)
    for path, meta in registry.items():
        name = Path(path).name
        print(f"{name:<50} {meta['source_label']:<30} {meta['chunks_added']:>6}  {meta['pages']}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Ingest PDFs into the petrophysics vectorstore.")
    parser.add_argument("files", nargs="*", help="PDF file(s) to ingest. Defaults to all in ./content/")
    parser.add_argument("--force",  action="store_true", help="Re-ingest even if already processed")
    parser.add_argument("--list",   action="store_true", help="List ingested files and exit")
    parser.add_argument("--pages",  nargs=2, type=int, metavar=("FIRST", "LAST"),
                        help="Page range for OCR, e.g. --pages 10 25")
    args = parser.parse_args()

    if args.list:
        list_ingested()
        return

    first_page = args.pages[0] if args.pages else 1
    last_page  = args.pages[1] if args.pages else None

    if args.files:
        pdf_paths = [Path(f) for f in args.files]
    else:
        pdf_paths = sorted(CONTENT_DIR.glob("*.pdf"))
        if not pdf_paths:
            print(f"No PDFs found in {CONTENT_DIR}. Pass file paths explicitly.")
            sys.exit(1)

    for pdf_path in pdf_paths:
        if not pdf_path.exists():
            print(f"[ERROR] File not found: {pdf_path}")
            continue
        print(f"\nIngesting: {pdf_path}")
        ingest_file(
            pdf_path,
            first_page=first_page,
            last_page=last_page,
            force=args.force,
        )

    print("\nAll done.")


if __name__ == "__main__":
    main()
