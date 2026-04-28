import pytest
import os
from services.vector_store import VectorStoreService
from services.embedding_service import EmbeddingService


@pytest.fixture
def vector_store_service():
    return VectorStoreService(EmbeddingService())

def test_get_collection_stats_raises_error(vector_store_service, monkeypatch):
    with monkeypatch.context() as m:

        def mock_count_fail(*args, **kwargs):
            raise Exception("Stats broken")

        m.setattr(
            vector_store_service.vector_store._collection, "count", mock_count_fail
        )

        with pytest.raises(Exception, match="Stats broken"):
            vector_store_service.get_collection_stats()

@pytest.mark.usefixtures("monkeypatch")
def test_add_texts_raises_error(monkeypatch, vector_store_service):
    def mock_add_texts_fail(*args, **kwargs):
        raise RuntimeError("Mock text error")

    monkeypatch.setattr(
        vector_store_service.vector_store, "add_texts", mock_add_texts_fail
    )

    with pytest.raises(RuntimeError, match="Mock text error"):
        vector_store_service.add_texts(["text"], [{}])

@pytest.mark.usefixtures("monkeypatch")
def test_similarity_search_raises_error(monkeypatch, vector_store_service):
    def mock_similarity_search(*args, **kwargs):
        raise Exception("Search failed")

    monkeypatch.setattr(
        vector_store_service.vector_store, "similarity_search", mock_similarity_search
    )

    with pytest.raises(Exception, match="Search failed"):
        vector_store_service.similarity_search("fail")

def test_clear_raises_error(tmp_path):
    broken_path = tmp_path / "protected"
    broken_path.mkdir()
    os.chmod(broken_path, 0o400)

    vs = VectorStoreService(EmbeddingService())
    vs.db_path = str(broken_path)

    with pytest.raises(Exception):
        vs.clear()

    os.chmod(broken_path, 0o700)

def test_get_retriever_raises_error(vector_store_service, monkeypatch):
    # Define the mock behavior
    def mock_as_retriever(*args, **kwargs):
        raise LookupError("Retriever error")

    # Apply monkeypatch in a limited context
    with monkeypatch.context() as m:
        m.setattr(vector_store_service.vector_store, "as_retriever", mock_as_retriever)

        # Test that the error is raised as expected
        with pytest.raises(LookupError, match="Retriever error"):
            vector_store_service.get_retriever()
