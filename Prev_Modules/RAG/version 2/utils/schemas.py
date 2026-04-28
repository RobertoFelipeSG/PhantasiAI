from pydantic import BaseModel, Field
from typing import Dict, Any, Optional
from datetime import datetime

class DocumentMetadata(BaseModel):
    """Schema for document metadata validation."""
    source_path: str
    file_type: str
    creation_date: datetime = Field(default_factory=datetime.now)
    last_modified: Optional[datetime] = None
    num_pages: Optional[int] = None
    word_count: Optional[int] = None
    chunk_index: Optional[int] = None
    total_chunks: Optional[int] = None
    extra: Dict[str, Any] = Field(default_factory=dict)
