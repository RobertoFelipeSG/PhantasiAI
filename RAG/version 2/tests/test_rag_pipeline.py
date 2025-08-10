from pathlib import Path
import pytest
import os
from services.rag_service import RAGService
from langchain_core.documents import Document
from services.vector_store import VectorStoreService
from services.embedding_service import EmbeddingService
from services.llm_service import LLMService

TEST_FILE_PATH = "tests/documents/what_is_rag.txt"


# NB : we may want to intialize a new vector database for testing purposes


@pytest.fixture
def rag_service():
    """Fixture to initialize the RAG service."""
    return RAGService()


@pytest.fixture
def vector_store_service():
    return VectorStoreService(EmbeddingService())


def test_init_vector_db(rag_service):
    """Fixture to initialize the vector database."""
    vector_store = rag_service.vector_store_service
    stats = vector_store.get_collection_stats()

    assert vector_store.vector_store is not None, "Vector store should be initialized."
    assert stats is not None, "Collection stats should not be None."
    assert os.path.exists(stats["db_path"]), "Vector DB directory does not exist"
    assert "all-MiniLM-L6-v2" in stats["embedding_function"], "Embedding model mismatch"
    assert stats["total_documents"] >= 0, "Total documents should be non-negative."


def test_add_new_document_to_vector_db(vector_store_service, rag_service):
    """Fixture to set up a test vector database."""

    # get initial embedding count
    initial_stats = vector_store_service.get_collection_stats()
    initial_count = initial_stats["total_documents"]

    rag_service.setup_vector_db(file_path=TEST_FILE_PATH)

    updated_stats = vector_store_service.get_collection_stats()
    updated_count = updated_stats["total_documents"]

    assert (
        updated_count > initial_count
    ), f"No new embeddings added. Initial: {initial_count}, After: {updated_count}"


def test_setup_vector_db_adds_documents(tmp_path: Path):
    test_file = tmp_path / "sample.txt"
    test_file.write_text("Hello world. This is a test document.")

    rag_service = RAGService()
    rag_service.setup_vector_db(str(test_file))

    stats = rag_service.vector_store_service.get_collection_stats()
    assert stats["total_documents"] > 0


def test_create_rag_chain_runs(rag_service):
    chain = rag_service.create_rag_chain()
    assert callable(chain.invoke)


def test_setup_vector_db_raises_error_on_missing_file():
    rag_service = RAGService()
    with pytest.raises(Exception):
        rag_service.setup_vector_db("non_existent.txt")


def test_add_documents_adds_data(vector_store_service):
    doc = Document(page_content="Test document content")
    vector_store_service.add_documents([doc])

    stats = vector_store_service.get_collection_stats()
    assert stats["total_documents"] > 0


def test_add_texts_adds_data(vector_store_service):
    vector_store_service.add_texts(["Test text"], [{"source": "unit_test"}])

    stats = vector_store_service.get_collection_stats()
    assert stats["total_documents"] > 0



def test_similarity_search(vector_store_service):
    vector_store_service.add_texts(["This is a similarity test"], [{}])
    results = vector_store_service.similarity_search("similarity")
    assert isinstance(results, list)
    assert len(results) > 0


def test_clear_removes_vector_store(tmp_path: Path):
    # Create fake DB path
    db_path = tmp_path / "fake_db"
    db_path.mkdir()

    vector_store = VectorStoreService(EmbeddingService())
    vector_store.db_path = str(db_path)
    vector_store.clear()

    assert not os.path.exists(db_path)



def test_get_retriever(vector_store_service):
    retriever = vector_store_service.get_retriever()
    assert retriever is not None


def test_llm_service():
    service = LLMService()

    llm = service.get_llm()
    prompt = service.get_prompt()

    # Prepare prompt input
    input_data = {
        "context": "LangChain is an open-source framework for building applications with LLMs.",
        "question": "What is LangChain?",
    }

    # Format the prompt
    full_prompt = prompt.invoke(input_data)

    # Invoke the actual model
    response = llm.invoke(full_prompt)

    # Assert something basic
    assert isinstance(response.content, str)
    assert (
        "LangChain" in response.content
        or "framework" in response.content
        or len(response.content) > 10
    )


def test_rag_pipeline_end_to_end(rag_service):
    """Test the end-to-end RAG pipeline."""

    rag_service.setup_vector_db("tests/documents/fake_client_info.txt")

    # Create RAG chain
    rag_chain = rag_service.create_rag_chain()

    # Prepare a test query
    query = "What is the client's phone?"

    # Invoke the chain
    response = rag_chain.invoke({"question": query})

    # Assert the response is valid
    assert isinstance(response, str)
    assert len(response) > 0
    # assert "james" in response.lower() or "parsons" in response.lower()
    assert "18779876403" in response.replace("-", "")
    print(f"Response: {response}")
