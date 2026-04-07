from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STORAGE_DIR = PROJECT_ROOT / "storage"
SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}
SKIP_DIRS = {"src", "storage", ".git", "__pycache__", ".venv", "venv"}
TOP_LEVEL_EXCLUDES = {"README.md", "requirements.txt"}
TOKEN_PATTERN = re.compile(r"\b[\w.+-]+\b", re.UNICODE)


@dataclass
class ChunkRecord:
    chunk_id: int
    source_path: str
    source_relpath: str
    page: int | None
    page_start: int | None
    page_end: int | None
    segment_index: int
    chunk_number: int
    title: str | None
    text: str


@dataclass
class IndexData:
    config: dict[str, Any]
    chunks: list[dict[str, Any]]
    embeddings: np.ndarray
    idf: np.ndarray


@dataclass
class SkippedFile:
    source_path: str
    source_relpath: str
    reason: str


@dataclass
class IngestResult:
    config: dict[str, Any]
    skipped_files: list[SkippedFile]


def ensure_pypdf():
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError(
            "PDF support requires the 'pypdf' package. Install the dependencies from "
            "requirements.txt before running ingestion."
        ) from exc
    return PdfReader


def normalize_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = text.replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_nonempty_lines(text: str) -> list[str]:
    return [normalize_text(line) for line in text.splitlines() if normalize_text(line)]


def word_count(text: str) -> int:
    return len(text.split())


def is_noise_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return True
    if re.fullmatch(r"[\d.\-_/ ]+", stripped):
        return True
    if re.fullmatch(r"page\s+\d+", stripped.lower()):
        return True
    return False


def looks_like_title(line: str) -> bool:
    words = line.split()
    if not words or len(words) > 18 or len(line) > 160:
        return False
    if re.fullmatch(r"[\W\d_]+", line):
        return False

    alpha_count = sum(character.isalpha() for character in line)
    digit_count = sum(character.isdigit() for character in line)
    if alpha_count < 3 or alpha_count < digit_count:
        return False
    if line.endswith(".") and len(words) > 10:
        return False
    return True


def extract_slide_title(text: str) -> tuple[str | None, str]:
    lines = split_nonempty_lines(text)
    if not lines:
        return None, ""

    title_index: int | None = None
    for index, line in enumerate(lines[:5]):
        if is_noise_line(line):
            continue
        if looks_like_title(line):
            title_index = index
            break

    if title_index is None:
        return None, "\n".join(lines)

    title = lines[title_index]
    body_lines = lines[:title_index] + lines[title_index + 1 :]
    return title, "\n".join(body_lines).strip()


def chunk_lines(lines: list[str], chunk_size: int, overlap: int) -> list[str]:
    if not lines:
        return []

    chunks: list[str] = []
    start = 0

    while start < len(lines):
        current_lines: list[str] = []
        current_words = 0
        end = start

        while end < len(lines):
            line = lines[end]
            line_words = max(1, word_count(line))
            if current_lines and current_words + line_words > chunk_size:
                break

            current_lines.append(line)
            current_words += line_words
            end += 1

            if current_words >= chunk_size:
                break

        chunk = "\n".join(current_lines).strip()
        if chunk:
            chunks.append(chunk)

        if end >= len(lines):
            break

        overlap_words = 0
        next_start = end
        while next_start > start and overlap_words < overlap:
            next_start -= 1
            overlap_words += max(1, word_count(lines[next_start]))

        if next_start <= start:
            next_start = max(end, start + 1)
        start = next_start

    return chunks


def chunk_segment_text(
    text: str,
    chunk_size: int,
    overlap: int,
    preserve_lines: bool,
) -> list[str]:
    if preserve_lines:
        lines = split_nonempty_lines(text)
        if not lines:
            return []
        if sum(word_count(line) for line in lines) <= chunk_size:
            return ["\n".join(lines)]
        return chunk_lines(lines, chunk_size=chunk_size, overlap=overlap)

    return chunk_text(text, chunk_size=chunk_size, overlap=overlap)


def build_embedding_text(chunk: ChunkRecord | dict[str, Any]) -> str:
    title = chunk.get("title") if isinstance(chunk, dict) else chunk.title
    text = chunk.get("text") if isinstance(chunk, dict) else chunk.text
    parts = [part for part in (title, text) if part]
    return "\n".join(parts).strip()


def scan_documents(root: Path) -> list[Path]:
    documents: list[Path] = []

    for current_root, dirnames, filenames in os.walk(root):
        dirnames[:] = [
            name for name in dirnames if name not in SKIP_DIRS and not name.startswith(".")
        ]

        current_path = Path(current_root)
        for filename in filenames:
            if filename.startswith("."):
                continue

            path = current_path / filename
            if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue

            relative_path = path.relative_to(root)
            if len(relative_path.parts) == 1 and filename in TOP_LEVEL_EXCLUDES:
                continue

            documents.append(path)

    documents.sort()
    return documents


def read_text_file(path: Path) -> str:
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue

    return path.read_text(encoding="utf-8", errors="ignore")


def extract_segments(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        PdfReader = ensure_pypdf()
        try:
            reader = PdfReader(str(path))
        except Exception as exc:
            raise RuntimeError(f"Failed to open PDF: {path}") from exc

        segments: list[dict[str, Any]] = []
        for page_number, page in enumerate(reader.pages, start=1):
            try:
                raw_text = page.extract_text() or ""
            except Exception as exc:
                raise RuntimeError(f"Failed to extract text from {path} page {page_number}") from exc

            text = normalize_text(raw_text)
            if text:
                title, body_text = extract_slide_title(text)
                segments.append(
                    {
                        "page": page_number,
                        "page_start": page_number,
                        "page_end": page_number,
                        "segment_index": page_number,
                        "title": title,
                        "text": body_text or text,
                        "preserve_lines": True,
                    }
                )

        return segments

    text = normalize_text(read_text_file(path))
    if not text:
        return []

    return [
        {
            "page": None,
            "page_start": None,
            "page_end": None,
            "segment_index": 1,
            "title": None,
            "text": text,
            "preserve_lines": False,
        }
    ]


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    words = text.split()
    if not words:
        return []

    if len(words) <= chunk_size:
        return [" ".join(words)]

    step = max(1, chunk_size - overlap)
    chunks: list[str] = []

    for start in range(0, len(words), step):
        end = min(len(words), start + chunk_size)
        chunk = " ".join(words[start:end]).strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(words):
            break

    return chunks


def build_chunk_records(
    root: Path,
    path: Path,
    chunk_size: int,
    overlap: int,
    next_chunk_id: int,
) -> list[ChunkRecord]:
    segments = extract_segments(path)
    records: list[ChunkRecord] = []
    relative_path = str(path.relative_to(root))

    for segment in segments:
        page = segment["page"]
        title = segment.get("title")
        segment_text = segment["text"]
        chunk_texts = chunk_segment_text(
            segment_text,
            chunk_size=chunk_size,
            overlap=overlap,
            preserve_lines=bool(segment.get("preserve_lines")),
        )
        if not chunk_texts:
            fallback_text = segment_text or title or ""
            if fallback_text:
                chunk_texts = [fallback_text]

        for chunk_number, chunk in enumerate(
            chunk_texts,
            start=1,
        ):
            records.append(
                ChunkRecord(
                    chunk_id=next_chunk_id,
                    source_path=str(path),
                    source_relpath=relative_path,
                    page=page,
                    page_start=segment.get("page_start"),
                    page_end=segment.get("page_end"),
                    segment_index=int(segment.get("segment_index", 1)),
                    chunk_number=chunk_number,
                    title=title,
                    text=chunk,
                )
            )
            next_chunk_id += 1

    return records


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


def token_to_dimension(token: str, dimensions: int) -> int:
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little") % dimensions


def build_tfidf_embeddings(
    chunks: list[ChunkRecord],
    dimensions: int,
) -> tuple[np.ndarray, np.ndarray]:
    vectors = np.zeros((len(chunks), dimensions), dtype=np.float32)
    document_frequency = np.zeros(dimensions, dtype=np.int32)

    for row_index, chunk in enumerate(chunks):
        hashed_counts: Counter[int] = Counter()
        for token in tokenize(build_embedding_text(chunk)):
            hashed_counts[token_to_dimension(token, dimensions)] += 1

        if not hashed_counts:
            continue

        for dimension, count in hashed_counts.items():
            vectors[row_index, dimension] = 1.0 + math.log(count)

        document_frequency[list(hashed_counts.keys())] += 1

    idf = np.log((1.0 + len(chunks)) / (1.0 + document_frequency)) + 1.0
    vectors *= idf.astype(np.float32)

    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    vectors /= norms

    return vectors.astype(np.float32), idf.astype(np.float32)


def vectorize_text(text: str, dimensions: int, idf: np.ndarray) -> np.ndarray:
    hashed_counts: Counter[int] = Counter()
    for token in tokenize(text):
        hashed_counts[token_to_dimension(token, dimensions)] += 1

    vector = np.zeros(dimensions, dtype=np.float32)
    if not hashed_counts:
        return vector

    for dimension, count in hashed_counts.items():
        vector[dimension] = 1.0 + math.log(count)

    vector *= idf
    norm = np.linalg.norm(vector)
    if norm > 0:
        vector /= norm

    return vector.astype(np.float32)


def save_index(
    storage_dir: Path,
    config: dict[str, Any],
    chunks: list[ChunkRecord],
    embeddings: np.ndarray,
    idf: np.ndarray,
) -> None:
    storage_dir.mkdir(parents=True, exist_ok=True)

    (storage_dir / "index_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (storage_dir / "chunks.json").write_text(
        json.dumps([asdict(chunk) for chunk in chunks], ensure_ascii=False),
        encoding="utf-8",
    )
    np.save(storage_dir / "embeddings.npy", embeddings)
    np.save(storage_dir / "idf.npy", idf)


def load_index(storage_dir: Path = DEFAULT_STORAGE_DIR) -> IndexData:
    config_path = storage_dir / "index_config.json"
    chunks_path = storage_dir / "chunks.json"
    embeddings_path = storage_dir / "embeddings.npy"
    idf_path = storage_dir / "idf.npy"

    missing_paths = [
        str(path)
        for path in (config_path, chunks_path, embeddings_path, idf_path)
        if not path.exists()
    ]
    if missing_paths:
        raise FileNotFoundError(
            "Index files are missing. Run 'python3 src/ingest.py' first.\n"
            + "\n".join(missing_paths)
        )

    config = json.loads(config_path.read_text(encoding="utf-8"))
    chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
    embeddings = np.load(embeddings_path)
    idf = np.load(idf_path)
    return IndexData(config=config, chunks=chunks, embeddings=embeddings, idf=idf)


def build_index(
    root: Path,
    storage_dir: Path,
    chunk_size: int,
    overlap: int,
    dimensions: int,
) -> IngestResult:
    documents = scan_documents(root)
    if not documents:
        raise RuntimeError(f"No supported documents were found under {root}")

    chunks: list[ChunkRecord] = []
    skipped_files: list[SkippedFile] = []
    next_chunk_id = 1
    indexed_files = 0

    for path in documents:
        try:
            chunk_records = build_chunk_records(
                root=root,
                path=path,
                chunk_size=chunk_size,
                overlap=overlap,
                next_chunk_id=next_chunk_id,
            )
        except Exception as exc:
            skipped_file = SkippedFile(
                source_path=str(path),
                source_relpath=str(path.relative_to(root)),
                reason=str(exc),
            )
            skipped_files.append(skipped_file)
            print(
                f"Skipping {skipped_file.source_relpath}: {skipped_file.reason}",
                file=sys.stderr,
            )
            continue

        if not chunk_records:
            continue

        chunks.extend(chunk_records)
        next_chunk_id = chunks[-1].chunk_id + 1
        indexed_files += 1

    if not chunks:
        raise RuntimeError(
            "No chunks were indexed. Supported files were found, but none could be processed "
            "into usable text."
        )

    embeddings, idf = build_tfidf_embeddings(chunks, dimensions=dimensions)
    config = {
        "indexed_at_utc": datetime.now(timezone.utc).isoformat(),
        "root_path": str(root),
        "storage_path": str(storage_dir),
        "supported_extensions": sorted(SUPPORTED_EXTENSIONS),
        "scanned_files": len(documents),
        "indexed_files": indexed_files,
        "skipped_files": len(skipped_files),
        "chunk_count": len(chunks),
        "chunk_size_words": chunk_size,
        "chunk_overlap_words": overlap,
        "embedding_dimensions": dimensions,
        "embedding_type": "hashed_tfidf",
    }
    save_index(
        storage_dir=storage_dir,
        config=config,
        chunks=chunks,
        embeddings=embeddings,
        idf=idf,
    )
    return IngestResult(config=config, skipped_files=skipped_files)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a local RAG index for study materials.")
    parser.add_argument(
        "--root",
        type=Path,
        default=PROJECT_ROOT,
        help="Folder to index recursively. Defaults to the project root.",
    )
    parser.add_argument(
        "--storage",
        type=Path,
        default=DEFAULT_STORAGE_DIR,
        help="Directory where the chunk metadata and embeddings are stored.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=220,
        help="Chunk size in words.",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=40,
        help="Overlap between chunks in words.",
    )
    parser.add_argument(
        "--dimensions",
        type=int,
        default=2048,
        help="Embedding dimensionality for the hashed TF-IDF vectors.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.chunk_overlap >= args.chunk_size:
        print("--chunk-overlap must be smaller than --chunk-size", file=sys.stderr)
        return 1

    try:
        result = build_index(
            root=args.root.resolve(),
            storage_dir=args.storage.resolve(),
            chunk_size=args.chunk_size,
            overlap=args.chunk_overlap,
            dimensions=args.dimensions,
        )
    except Exception as exc:
        print(f"Ingestion failed: {exc}", file=sys.stderr)
        return 1

    config = result.config
    print(
        f"Scanned {config['scanned_files']} supported files.\n"
        f"Indexed {config['indexed_files']} files into {config['chunk_count']} chunks.\n"
        f"Skipped {config['skipped_files']} files.\n"
        f"Saved index to {config['storage_path']}"
    )
    if result.skipped_files:
        print("\nSkipped files:")
        for skipped_file in result.skipped_files:
            print(f"- {skipped_file.source_relpath}: {skipped_file.reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
