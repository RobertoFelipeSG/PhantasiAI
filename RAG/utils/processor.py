from typing import List
import os
import logging
from langchain_core.documents import Document
from .document_loader import DocumentLoader
from .text_splitter import TextSplitter

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Main interface for document processing pipeline."""

    def __init__(
            self,
            chunk_size: int = None,
            chunk_overlap: int = None,
            splitting_strategy: str = 'recursive'
    ):
        self.loader = DocumentLoader()
        self.splitter = TextSplitter(chunk_size, chunk_overlap)
        self.splitting_strategy = splitting_strategy

    def process_document(self, file_path: str) -> List[Document]:
        """Process a single document through the complete pipeline."""
        try:
            documents = self.loader.load_document(file_path)
            split_documents = self.splitter.split_documents(
                documents,
                strategy=self.splitting_strategy
            )
            return split_documents

        except Exception as e:
            logger.error(f"Error processing document {file_path}: {e}")
            raise

    def process_directory(self, directory_path: str) -> List[Document]:
        """Process all supported documents in a directory."""
        try:
            all_documents = []
            supported_extensions = tuple(self.loader.loaders.keys())

            for root, _, files in os.walk(directory_path):
                for file in files:
                    if file.lower().endswith(supported_extensions):
                        file_path = os.path.join(root, file)
                        documents = self.process_document(file_path)
                        all_documents.extend(documents)

            logger.info(
                f"Processed {len(all_documents)} total chunks from "
                f"directory {directory_path}"
            )
            return all_documents

        except Exception as e:
            logger.error(f"Error processing directory {directory_path}: {e}")
            raise
