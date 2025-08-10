from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.runnable import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import TextLoader
from core.config import settings
from services.embedding_service import EmbeddingService
from services.llm_service import LLMService
from services.vector_store import VectorStoreService
import os
import logging
from typing import Union
from pathlib import Path

logger = logging.getLogger(__name__)


class RAGService:
    def __init__(self, search_type: str = "similarity"):
        self.embedding_service = EmbeddingService()
        self.llm_service = LLMService()
        self.vector_store_service = VectorStoreService(self.embedding_service)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            separators=["\n\n", "\n", ".", "!", "?", ";", ":", " ", ""],
        )
        self.search_type = search_type

    def setup_vector_db(self, file_paths: Union[str, list[str]]) -> None:
        """Initialize and populate vector database."""
        if isinstance(file_paths, str):
            file_paths = [file_paths]

        all_split_docs = []
        for file_path in file_paths:
            try:
                # Create database directory if it doesn't exist
                os.makedirs(settings.DB_DIR, exist_ok=True)

                # Load and split documents
                loader = TextLoader(file_path, encoding="utf-8")
                documents = loader.load()
                for d in documents:
                    d.metadata["source"] = Path(file_path).name
                    d.metadata["doc_id"] = Path(file_path).name
                
                # propagate metadata while splitting
                split = self.text_splitter.split_documents(documents)
                for idx, chunk in enumerate(split):
                    chunk.metadata["chunk_id"] = idx
                all_split_docs.extend(split)


                # Add to vector store
                self.vector_store_service.add_documents(all_split_docs)

                # Log stats
                stats = self.vector_store_service.get_collection_stats()
                logger.info(
                    f"Vector store initialized with {stats['total_documents']} documents"
                )

            except Exception as e:
                logger.error(f"Error setting up vector database: {e}")
                raise

    def create_rag_chain(self):
        """Create the RAG chain for retrieval and response generation."""
        try:
            retriever = self.vector_store_service.get_retriever(
                search_type="similarity", k=settings.TOP_K
            )

            def format_docs(docs):
                """Group retrieved chunks by PDF and mark them in plain text."""
                grouped = {}
                for d in docs:
                    src = d.metadata.get("source", "UNKNOWN_SOURCE")
                    grouped.setdefault(src, []).append(d.page_content)

                blocks = []
                for src, pages in grouped.items():
                    blocks.append(f"### SOURCE: {src}\n" + "\n".join(pages))

                return "\n\n".join(blocks)

            chain = (
                {"context": retriever | format_docs, "question": RunnablePassthrough()}
                | self.llm_service.get_prompt()
                | self.llm_service.get_llm()
                | StrOutputParser()
            )

            return chain

        except Exception as e:
            logger.error(f"Error creating RAG chain: {e}")
            raise
