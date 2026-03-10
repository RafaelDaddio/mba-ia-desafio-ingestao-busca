import os
from dotenv import load_dotenv


from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document
from langchain_postgres import PGVector
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

PDF_PATH = os.getenv("PDF_PATH")


def clean_text(text: str) -> str:
    return text.replace("\x00", "") if text else text


def create_splits(pages: list[Document]) -> list[Document]:
    splits = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        add_start_index=False).split_documents(pages)
    if not splits:
        raise SystemExit(0)

    return [
        Document(
            page_content=clean_text(d.page_content),
            metadata={k: v for k, v in d.metadata.items()
                      if v not in (None, "")},
        ) for d in splits
    ]


def ingest_pdf():
    pages = PyPDFLoader(PDF_PATH).load()

    enriched_splits = create_splits(pages)

    ids = [f"doc-{i}" for i in range(len(enriched_splits))]

    embeddings = OpenAIEmbeddings(
        model=os.getenv("OPENAI_MODEL", "text-embedding-3-small")
    )

    store = PGVector(
        embeddings=embeddings,
        collection_name=os.getenv("PG_VECTOR_COLLECTION_NAME"),
        connection=os.getenv("DATABASE_URL"),
        use_jsonb=True
    )

    store.add_documents(
        documents=enriched_splits,
        ids=ids
    )


if __name__ == "__main__":
    ingest_pdf()
