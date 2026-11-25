
import logging
import hashlib
from pathlib import Path

from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import InputFormat
from docling.chunking import HybridChunker
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer


from transformers import AutoTokenizer

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


# =====================================
# CONFIGURATION
# =====================================

MD_DIR = Path("./data/extracted_md")
DB_DIR = "./data/chroma_db"

COLLECTION_NAME = "RuleOut"

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

MAX_TOKENS = 512


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =====================================
# INITIALIZE CHUNKER
# =====================================

hf_tokenizer = AutoTokenizer.from_pretrained(
    EMBEDDING_MODEL
)

docling_tokenizer = HuggingFaceTokenizer(
    tokenizer=hf_tokenizer
)

chunker = HybridChunker(
    tokenizer=docling_tokenizer,
    max_tokens=MAX_TOKENS,
    merge_peers=True
)


# =====================================
# INITIALIZE EMBEDDINGS
# =====================================

embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={
        "device": "cpu"
    },
    encode_kwargs={
        "normalize_embeddings": True
    }
)


# =====================================
# INITIALIZE VECTOR DATABASE
# =====================================

vector_store = Chroma(
    collection_name=COLLECTION_NAME,
    persist_directory=DB_DIR,
    embedding_function=embeddings
)


# =====================================
# LOAD MARKDOWN
# =====================================

def load_markdown_files():

    files = sorted(MD_DIR.rglob("*.md"))

    documents = []

    for file_path in files:

        text = file_path.read_text(
            encoding="utf-8"
        )

        if not text.strip():
            continue

        documents.append({
            "source": file_path.name,
            "path": str(file_path),
            "text": text
        })

    logger.info(
        "Loaded %s Markdown files",
        len(documents)
    )

    return documents


# =====================================
# CONVERT MARKDOWN TO DOCLING DOCUMENT
# =====================================

def markdown_to_docling(markdown_text):

    converter = DocumentConverter()

    result = converter.convert_string(
        markdown_text,
        format=InputFormat.MD
    )

    return result.document


# =====================================
# CHUNK DOCUMENTS
# =====================================

def chunk_documents(documents):

    all_chunks = []

    for document in documents:

        logger.info(
            "Chunking %s",
            document["source"]
        )

        docling_document = markdown_to_docling(
            document["text"]
        )

        chunks = chunker.chunk(
            docling_document
        )

        for index, chunk in enumerate(chunks):

            content = chunker.contextualize(chunk)

            if not content.strip():
                continue

            chunk_id = hashlib.sha256(
                (
                    document["path"]
                    + str(index)
                    + content
                ).encode("utf-8")
            ).hexdigest()

            metadata = {
                "source_file": document["source"],
                "source_path": document["path"],
                "chunk_index": index,
                "chunk_id": chunk_id,
                "headings": " > ".join(
                    getattr(chunk.meta, "headings", []) or []
                )
            }

            all_chunks.append(
                Document(
                    page_content=content,
                    metadata=metadata
                )
            )

    logger.info(
        "Generated %s chunks",
        len(all_chunks)
    )

    return all_chunks


# =====================================
# STORE IN CHROMA
# =====================================


def store_chunks(chunks, batch_size=256):

    if not chunks:
        logger.warning("No chunks to store")
        return

    total = len(chunks)

    for start in range(0, total, batch_size):

        batch = chunks[start:start + batch_size]

        ids = [
            chunk.metadata["chunk_id"]
            for chunk in batch
        ]

        vector_store.add_documents(
            documents=batch,
            ids=ids
        )

        completed = min(start + batch_size, total)

        logger.info(
            "Stored %d / %d chunks in Chroma",
            completed,
            total
        )

    logger.info(
        "Successfully stored all %d chunks!",
        total
    )


# =====================================
# MAIN PIPELINE
# =====================================

def main():

    documents = load_markdown_files()

    if not documents:
        logger.warning("No Markdown files found")
        return

    chunks = chunk_documents(documents)

    store_chunks(chunks)

    logger.info("Knowledge base ingestion complete!")


if __name__ == "__main__":
    main()
