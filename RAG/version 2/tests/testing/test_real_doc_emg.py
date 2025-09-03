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
            "What does the paper say EMG recording represent?",
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
    
    """Debug: print top retrieved chunks"""
    k = 5
    docs = vector_store_service.similarity_search(query, k=k) # old: docs = vector_store_service.similarity_search(query, k=1)
    
    print(f"\n=== DEBUG: Query ===\n{query}")
    print(f"Expected snippet: {expected_snippet}\n")
    print("--- Retrieved chunks ---")
    for i, d in enumerate(docs, start=1):
        preview = d.page_content.strip().replace("\n", " ")
        print(f"[{i}] {preview[:300]}{'...' if len(preview) > 300 else ''}")
    
    """Top-1 chunk returned by similarity_search should mention the snippet."""
    assert len(docs) > 0 # old: assert len(docs) == 1
    top_chunk = docs[0].page_content.lower()

    if expected_snippet.lower() not in top_chunk:
        print("Expected snippet not found in Top-1 chunk")

        for d in docs:
            if expected_snippet.lower() in d.page_content.lower():
                print("Expected snippet found in a lower-ranked chunk")
                break
        else: print("Expected snippet not found in any retrieved chunks (Top-5)")

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
    """Debug: print retrieved content and LLM answer"""
    retriever = rag_service.vector_store_service.get_retriever(search_type="similarity", k=5)

    # context passed to LLM
    retrieved_docs = retriever.invoke(question)
    print(f"\n=== DEBUG: Question ===\n{question}\n")
    print("--- Retrieved context chunks ---")
    for i, d in enumerate(retrieved_docs, start=1):
        preview = d.page_content.strip().replace("\n", " ")
        print(f"[{i}] {preview[:300]}{'...' if len(preview) > 300 else ''}")
    
    """Full RAG chain should return an answer that references expected terms."""
    rag_chain = rag_service.create_rag_chain()
    response = rag_chain.invoke({"question": question})
    answer = response if isinstance(response, str) else response.content

    print(f"\n--- LLM Answer ---\n{answer}")

    answer_lower = answer.lower()

    # Check for specific keywords missing (if any)
    missing = [kw for kw in keywords if kw not in answer_lower]
    if missing:
        print(f"\nMissing keywords: {missing}")

    # Basic sanity: we got a non-empty answer
    assert answer_lower.strip() != ""
    # Soft check: at least 1 keyword appears
    assert any(
        kw in answer_lower for kw in keywords
    ), f"None of the expected keywords {keywords} found in:\n{answer}"