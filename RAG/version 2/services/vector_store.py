from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain.retrievers import ContextualCompressionRetriever
from typing import List, Optional, Dict, Any
from core.config import settings
from utils.reranker import RerankingRetriever
import logging
import os
import shutil

import time

logger = logging.getLogger(__name__)

class VectorStoreService:
    print("DEBUG: Initializing VectorStore")

    def __init__(self, embedding_service):
        """Initialize vector store service with embedding model."""
        self.embedding_function = embedding_service.get_embeddings()
        self.db_path = settings.DB_DIR
        self.distance_metric = settings.CHROMA_DISTANCE_METRIC
        self._vector_store = None

    @property
    def vector_store(self) -> Chroma:
        """Lazy loading of vector store."""
        if self._vector_store is None:
            metadata = {"hnsw:space": self.distance_metric}
            
            if os.path.exists(self.db_path):
                self._vector_store = Chroma(
                    persist_directory=self.db_path,
                    embedding_function=self.embedding_function,
                    collection_metadata=metadata
                )
                logger.info("Loaded existing vector store")
            else:
                self._vector_store = Chroma(
                    embedding_function=self.embedding_function,
                    persist_directory=self.db_path
                )
                logger.info("Created new vector store")
        return self._vector_store

    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to vector store."""
        start = time.time()
        try:
            self.vector_store.add_documents(documents)
            print(f"DEBUG: Added docs took {time.time()-start:.2f}s")
            logger.info(f"Added {len(documents)} documents to vector store")
        except Exception as e:
            logger.error(f"Error adding documents to vector store: {e}")
            raise

    def add_texts(self, texts: List[str], metadatas: Optional[List[Dict[str, Any]]] = None, 
                  embeddings: Optional[List[List[float]]] = None) -> None:
        """Add raw texts to vector store."""
        try:
            if embeddings is None: # fall back to computing embeddings
                embeddings = self.embedding_function.embed_documents(texts)
            
            self.vector_store.add_texts(texts=texts, metadatas=metadatas, embeddings=embeddings)
            logger.info(f"Added {len(texts)} texts to vector store")
        except Exception as e:
            logger.error(f"Error adding texts to vector store: {e}")
            raise

    def similarity_search(self, query: str, k: int = None) -> List[Document]:
        """Perform similarity search."""
        try:
            k = k or settings.TOP_K

            # query is embedded internally -> underlying DB runs similarly search by vector
            results = self.vector_store.similarity_search(query, k=k)
            logger.debug(f"Retrieved {len(results)} documents for query: {query}")
            return results
        except Exception as e:
            logger.error(f"Error performing similarity search: {e}")
            raise

    def get_retriever(self, search_type: str = "similarity", rerank: bool = False, **kwargs):
        """Get a retriever interface to the vector store."""
        try:
            base_retriever = self.vector_store.as_retriever(
                search_type=search_type,
                search_kwargs=kwargs
            )
            if rerank:
                return RerankingRetriever(base_retriever)
            
            return base_retriever
        except Exception as e:
            logger.error(f"Error creating retriever: {e}")
            raise

    def clear(self) -> None:
        """Clear all documents from vector store."""
        try:
            if os.path.exists(self.db_path):
                shutil.rmtree(self.db_path)
            self._vector_store = None
            logger.info("Vector store cleared")
        except Exception as e:
            logger.error(f"Error clearing vector store: {e}")
            raise

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store."""
        try:
            collection = self.vector_store._collection
            return {
                "total_documents": collection.count(),
                "db_path": self.db_path,
                "embedding_function": getattr(self.embedding_function, 'model_name', 'Unknown'),
            }
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            raise
