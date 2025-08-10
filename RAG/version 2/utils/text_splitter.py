from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    MarkdownTextSplitter,
)
from langchain_core.documents import Document
from typing import List
import logging
from ..core.config import settings

logger = logging.getLogger(__name__)

class TextSplitter:
    """Configurable document chunking strategies."""

    def __init__(self, chunk_size: int = None, chunk_overlap: int = None):
        self.chunk_size = chunk_size or settings.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP

        self.splitters = {
            'recursive': RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separators=["\n\n", "\n", ".", "!", "?", ";", ":", " ", ""]
            ),
            'markdown': MarkdownTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )
        }

    def split_documents(
        self,
        documents: List[Document],
        strategy: str = 'recursive'
    ) -> List[Document]:
        """Split documents using specified strategy."""
        try:
            if strategy not in self.splitters:
                raise ValueError(f"Unsupported splitting strategy: {strategy}")

            splitter = self.splitters[strategy]
            split_docs = splitter.split_documents(documents)

            for idx, doc in enumerate(split_docs):
                doc.metadata.update({
                    "chunk_index": idx,
                    "total_chunks": len(split_docs)
                })

            logger.info(
                f"Split {len(documents)} documents into {len(split_docs)} chunks "
                f"using {strategy} strategy"
            )
            return split_docs

        except Exception as e:
            logger.error(f"Error splitting documents: {e}")
            raise
