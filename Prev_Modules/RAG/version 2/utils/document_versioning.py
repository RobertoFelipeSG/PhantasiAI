import os
import json
import hashlib
import shutil
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
from pydantic import BaseModel
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

class VersionMetadata(BaseModel):
    """Schema for document version metadata."""
    version_id: str
    source_path: str
    hash: str
    timestamp: datetime
    user: Optional[str] = None
    description: Optional[str] = None
    size_bytes: int
    chunk_count: Optional[int] = None
    processing_config: Dict[str, Any] = {}
    parent_version: Optional[str] = None

class DocumentVersion:
    """System for tracking document versions and changes."""

    def __init__(self, version_store_path: str = "./version_store"):
        self.version_store_path = version_store_path
        self.metadata_path = os.path.join(version_store_path, "metadata")
        self.content_path = os.path.join(version_store_path, "content")

        # Create directory structure if it doesn't exist
        os.makedirs(self.metadata_path, exist_ok=True)
        os.makedirs(self.content_path, exist_ok=True)

    def _compute_hash(self, file_path: str) -> str:
        """Compute SHA-256 hash of a file."""
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()

    def _generate_version_id(self, file_path: str, timestamp: datetime) -> str:
        """Generate a unique version ID based on file name and timestamp."""
        base_name = os.path.basename(file_path)
        time_str = timestamp.strftime("%Y%m%d%H%M%S")
        name_hash = hashlib.md5(base_name.encode()).hexdigest()[:8]
        return f"{name_hash}_{time_str}"

    def create_version(
        self,
        file_path: str,
        description: Optional[str] = None,
        user: Optional[str] = None,
        processing_config: Optional[Dict[str, Any]] = None,
        processed_chunks: Optional[List[Document]] = None,
        parent_version: Optional[str] = None
    ) -> VersionMetadata:
        """Create a new version of a document.

        Args:
            file_path: Path to the document to version
            description: Optional description of this version
            user: Optional username of person creating the version
            processing_config: Configuration used for processing
            processed_chunks: Processed chunks if already available
            parent_version: ID of parent version if this is a derivative

        Returns:
            VersionMetadata for the created version
        """
        timestamp = datetime.now()
        file_hash = self._compute_hash(file_path)
        version_id = self._generate_version_id(file_path, timestamp)

        # Create version content directory
        version_content_path = os.path.join(self.content_path, version_id)
        os.makedirs(version_content_path, exist_ok=True)

        # Copy original file
        file_name = os.path.basename(file_path)
        original_file_copy = os.path.join(version_content_path, f"original_{file_name}")
        shutil.copy2(file_path, original_file_copy)

        # Store processed chunks if available
        chunk_count = None
        if processed_chunks:
            chunk_count = len(processed_chunks)
            chunks_file = os.path.join(version_content_path, "processed_chunks.json")
            with open(chunks_file, 'w') as f:
                json.dump(
                    [
                        {
                            "content": doc.page_content,
                            "metadata": doc.metadata
                        }
                        for doc in processed_chunks
                    ],
                    f,
                    default=str  # Handle datetime serialization
                )

        # Create metadata
        metadata = VersionMetadata(
            version_id=version_id,
            source_path=os.path.abspath(file_path),
            hash=file_hash,
            timestamp=timestamp,
            user=user,
            description=description,
            size_bytes=os.path.getsize(file_path),
            chunk_count=chunk_count,
            processing_config=processing_config or {},
            parent_version=parent_version
        )

        # Save metadata
        metadata_file = os.path.join(self.metadata_path, f"{version_id}.json")
        with open(metadata_file, 'w') as f:
            f.write(metadata.model_dump_json(indent=2))

        logger.info(f"Created new document version: {version_id} for {file_path}")
        return metadata

    def get_version(self, version_id: str) -> Optional[VersionMetadata]:
        """Retrieve version metadata by ID."""
        metadata_file = os.path.join(self.metadata_path, f"{version_id}.json")
        if not os.path.exists(metadata_file):
            return None

        with open(metadata_file, 'r') as f:
            return VersionMetadata.model_validate_json(f.read())

    def get_versions_by_path(self, file_path: str) -> List[VersionMetadata]:
        """Get all versions of a specific document by its path."""
        abs_path = os.path.abspath(file_path)
        versions = []

        for metadata_file in os.listdir(self.metadata_path):
            if not metadata_file.endswith('.json'):
                continue

            try:
                with open(os.path.join(self.metadata_path, metadata_file), 'r') as f:
                    metadata = VersionMetadata.model_validate_json(f.read())
                    if metadata.source_path == abs_path:
                        versions.append(metadata)
            except Exception as e:
                logger.error(f"Error reading metadata file {metadata_file}: {e}")

        # Sort by timestamp, newest first
        return sorted(versions, key=lambda x: x.timestamp, reverse=True)

    def get_version_content(self, version_id: str) -> Optional[str]:
        """Get path to the original content of a specific version."""
        version_content_path = os.path.join(self.content_path, version_id)
        if not os.path.exists(version_content_path):
            return None

        # Find the original file in the version directory
        for file in os.listdir(version_content_path):
            if file.startswith("original_"):
                return os.path.join(version_content_path, file)

        return None

    def get_version_chunks(self, version_id: str) -> Optional[List[Document]]:
        """Get processed chunks for a specific version."""
        chunks_file = os.path.join(self.content_path, version_id, "processed_chunks.json")
        if not os.path.exists(chunks_file):
            return None

        with open(chunks_file, 'r') as f:
            chunks_data = json.load(f)

        # Convert back to Document objects
        documents = []
        for chunk in chunks_data:
            documents.append(
                Document(
                    page_content=chunk["content"],
                    metadata=chunk["metadata"]
                )
            )

        return documents

    def compare_versions(self, version_id1: str, version_id2: str) -> Dict[str, Any]:
        """Compare two versions of a document.

        Returns basic metadata comparison. For more sophisticated diff,
        external tools would be needed based on document type.
        """
        v1 = self.get_version(version_id1)
        v2 = self.get_version(version_id2)

        if not v1 or not v2:
            missing = []
            if not v1:
                missing.append(version_id1)
            if not v2:
                missing.append(version_id2)
            raise ValueError(f"Versions not found: {', '.join(missing)}")

        # Check if they're versions of the same document
        if Path(v1.source_path).name != Path(v2.source_path).name:
            logger.warning(f"Comparing different documents: {v1.source_path} vs {v2.source_path}")

        time_diff = abs((v1.timestamp - v2.timestamp).total_seconds())
        size_diff = v1.size_bytes - v2.size_bytes
        is_same_content = v1.hash == v2.hash

        return {
            "versions": {
                "v1": v1.version_id,
                "v2": v2.version_id,
            },
            "source_paths": {
                "v1": v1.source_path,
                "v2": v2.source_path,
            },
            "is_same_content": is_same_content,
            "time_difference_seconds": time_diff,
            "size_difference_bytes": size_diff,
            "size_change_percentage": (size_diff / v2.size_bytes) * 100 if v2.size_bytes > 0 else 0,
            "chunk_counts": {
                "v1": v1.chunk_count,
                "v2": v2.chunk_count,
            }
        }
