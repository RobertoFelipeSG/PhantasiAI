from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader
)
from langchain_core.documents import Document
from typing import List
import os
import logging
from .text_processing import TextProcessor
from .schemas import DocumentMetadata

logger = logging.getLogger(__name__)

class DocumentLoader:
    """Multi-format document loader."""

    def __init__(self):
        self.preprocessor = TextProcessor()
        self.loaders = {
            '.pdf': PyPDFLoader,
            '.docx': Docx2txtLoader,
            '.txt': TextLoader,
        }

    def load_document(self, file_path: str) -> List[Document]:
        """Load document based on file extension."""
        try:
            file_ext = os.path.splitext(file_path)[1].lower()
            if file_ext not in self.loaders:
                raise ValueError(f"Unsupported file type: {file_ext}")

            loader_class = self.loaders[file_ext]
            loader = loader_class(file_path)
            documents = loader.load()

            base_metadata = self.preprocessor.extract_metadata(file_path)

            cleaned_docs = []
            for idx, doc in enumerate(documents):
                cleaned_text = self.preprocessor.clean_text(doc.page_content)
                metadata = {
                    **base_metadata,
                    "chunk_index": idx,
                    "total_chunks": len(documents),
                    "word_count": len(cleaned_text.split())
                }
                if hasattr(doc.metadata, 'page'):
                    metadata['page'] = doc.metadata.page

                validated_metadata = DocumentMetadata(**metadata).model_dump()

                cleaned_docs.append(
                    Document(
                        page_content=cleaned_text,
                        metadata=validated_metadata
                    )
                )

            logger.info(f"Successfully loaded {len(cleaned_docs)} pages from {file_path}")
            return cleaned_docs

        except Exception as e:
            logger.error(f"Error loading document {file_path}: {e}")
            raise
