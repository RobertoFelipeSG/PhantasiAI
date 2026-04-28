import re
import os
from datetime import datetime
from typing import Dict, Any
import logging
from typing import List, Dict, Any
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

class TextProcessor:
    @staticmethod
    def clean_text(text: str) -> str:
        """Clean and normalize text."""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s.,!?-]', '', text)
        return text

    @staticmethod
    def extract_metadata(file_path: str) -> Dict[str, Any]:
        """Extract metadata from text content."""
        stats = os.stat(file_path)
        metadata = {
            # "char_count": len(text),
            # "word_count": len(text.split()),
            # "has_code": bool(re.search(r'```[\s\S]*?```', text))
            "source_path": file_path,
            "file_type": os.path.splitext(file_path)[1].lower(),
            "creation_date": datetime.fromtimestamp(stats.st_ctime),
            "last_modified": datetime.fromtimestamp(stats.st_mtime)
        }
        return metadata

    @staticmethod
    def preprocess_documents(documents: List[Document]) -> List[Document]:
        """Preprocess a list of documents."""
        processed_docs = []
        for doc in documents:
            cleaned_text = TextProcessor.clean_text(doc.page_content)
            metadata = {
                **doc.metadata,
                **TextProcessor.extract_metadata(cleaned_text)
            }
            processed_docs.append(
                Document(page_content=cleaned_text, metadata=metadata)
            )
        return processed_docs
