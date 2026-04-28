import pytest
import os
from app.services.rag_service import RAGService
from app.core.config import settings

@pytest.fixture
def sample_text():
    return """
    Python is a high-level programming language.
    It was created by Guido van Rossum.
    Python is known for its simplicity and readability.
    """

@pytest.fixture
def test_file(tmp_path, sample_text):
    file_path = tmp_path / "test.txt"
    with open(file_path, "w") as f:
        f.write(sample_text)
    return str(file_path)

@pytest.fixture
def rag_service():
    return RAGService()
