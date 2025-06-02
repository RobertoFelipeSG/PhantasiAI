from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.runnable import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import TextLoader
from ..core.config import settings
from .embedding_service import EmbeddingService
from .llm_service import LLMService
from .vector_store import VectorStoreService
import os
import logging

logger = logging.getLogger(__name__)

class RAGService:
    def __init__(self):
        self.embedding_service = EmbeddingService()
        self.llm_service = LLMService()
        self.vector_store_service = VectorStoreService(self.embedding_service)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            separators=["\n\n", "\n", ".", "!", "?", ";", ":", " ", ""]
        )

    def setup_vector_db(self, file_path: str) -> None:
        """Initialize and populate vector database."""
        try:
            # Create database directory if it doesn't exist
            os.makedirs(settings.DB_DIR, exist_ok=True)

            # Load and split documents
            loader = TextLoader(file_path)
            documents = loader.load()
            split_docs = self.text_splitter.split_documents(documents)

            # Add to vector store
            self.vector_store_service.add_documents(split_docs)

            # Log stats
            stats = self.vector_store_service.get_collection_stats()
            logger.info(f"Vector store initialized with {stats['total_documents']} documents")

        except Exception as e:
            logger.error(f"Error setting up vector database: {e}")
            raise

    def create_rag_chain(self):
        """Create the RAG chain for retrieval and response generation."""
        try:
            retriever = self.vector_store_service.get_retriever(
                search_type="similarity",
                k=settings.TOP_K
            )

            def format_docs(docs):
                return "\n\n".join(doc.page_content for doc in docs)

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
