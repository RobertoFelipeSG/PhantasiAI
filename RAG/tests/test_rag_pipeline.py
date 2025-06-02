import pytest
import os
from app.services.rag_service import setup_vector_db, retrieve_documents

@pytest.fixture
def vector_db():
    """Fixture to set up a test vector database."""
    # Get the absolute path to the test data file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    test_file_path = os.path.join(current_dir, "test_data", "test_doc.txt")

    # Make sure we're using the test file path
    return setup_vector_db(file_path=test_file_path)

def test_retrieval(vector_db):
    """Test that retrieval returns at least one document."""
    query = "What is RAG?"
    results = retrieve_documents(query, top_k=2)
    assert len(results) > 0, "No documents retrieved!"
