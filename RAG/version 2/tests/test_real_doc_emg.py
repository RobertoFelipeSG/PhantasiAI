"""
tests/test_real_doc_emg.py
Extra integration tests that use the Nature Biomedical Engineering
Perspective “Peripheral neural interfaces for reading high-frequency
brain signals”.

These tests assume the article’s raw text is stored at
tests/documents/nature_emg_perspective.txt
- N.B. Implementing conversion module will be necessary to handle PDFs
"""

import os
from pathlib import Path

import pytest
from langchain_core.documents import Document

from services.rag_service import RAGService
from services.vector_store import VectorStoreService
from services.embedding_service import EmbeddingService

TEST_EMG_FILE = "tests/documents/Ib--ez_et_al-2025-Nature_Biomedical_Engineering.txt"


@pytest.fixture(scope="module")
def rag_service():
    """Fixture to initialize the RAG service."""
    return RAGService()


@pytest.fixture(scope="module")
def vector_store_service(rag_service) -> VectorStoreService:
    """Shortcut to the inner VectorStoreService."""
    return rag_service.vector_store_service


def test_emg_doc_ingested(
    rag_service: RAGService, vector_store_service: VectorStoreService
) -> RAGService:
    """
    RAG service pre-loaded with the EMG article so every test
    in this module re-uses the same temporary vector-DB.
    """
    # start with a clean slate
    vector_store_service.clear()
    stats_before = rag_service.vector_store_service.get_collection_stats()

    # ingest the EMG article
    rag_service.setup_vector_db(TEST_EMG_FILE)
    stats_after = rag_service.vector_store_service.get_collection_stats()

    assert stats_before["total_documents"] == 0  # tests if VDB was cleared
    assert stats_after["total_documents"] > 0  # tests if the article was ingestd
    assert stats_after["total_documents"] >= 50  # tests if the article was chunked


@pytest.mark.parametrize(
    ("query", "expected_snippet"),
    [
        (
            "What does the paper say about EMG recordings?",
            "EMG recordings represent a mixture of the electrical activity",
        ),
        (
            "Explain how high-frequency inputs reach motor neurons",
            "high-frequency inputs may be associated with ongoing activity",
        ),
    ],
)
def test_direct_similarity_search(
    vector_store_service: VectorStoreService, query: str, expected_snippet: str
):
    """Top-1 chunk returned by similarity_search should mention the snippet."""
    docs = vector_store_service.similarity_search(query, k=1)
    assert len(docs) == 1
    top_chunk = docs[0].page_content.lower()
    assert expected_snippet.lower() in top_chunk


@pytest.mark.parametrize(
    ("question", "keywords"),
    [
        (
            "Summarize how the authors use EMG to decode spinal motor neuron activity.",
            ["emg", "motor unit", "decode"],
        ),
        (
            "Does the article discuss non-invasive neuromodulation or stimulation artefacts?",
            ["non-invasive", "stimulation", "artefact"],
        ),
        (
            "What bandwidth (frequency range) do they consider ‘high-frequency’ inputs?",
            [">10 hz", "beta", "gamma"],
        ),
    ],
)
def test_rag_chain_answers_emg_questions(
    rag_service: RAGService, question: str, keywords: list[str]
):
    """Full RAG chain should return an answer that references expected terms."""
    rag_chain = rag_service.create_rag_chain()
    response = rag_chain.invoke({"question": question})
    answer = response if isinstance(response, str) else response.content
    answer_lower = answer.lower()

    # Basic sanity: we got a non-empty answer
    assert answer_lower.strip() != ""
    # Soft check: at least 1 keyword appears
    assert any(
        kw in answer_lower for kw in keywords
    ), f"None of the expected keywords {keywords} found in:\n{answer}"