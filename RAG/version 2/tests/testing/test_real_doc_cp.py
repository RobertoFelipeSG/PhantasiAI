"""
tests/test_real_doc_cp.py
Extra integration tests that use "CPA Parents Guide"

These tests assume the article’s raw text is stored at
tests/documents/CPA52_13-17_ParentsGuide_WEB.txt
- N.B. Implementing conversion module will be necessary to handle PDFs
"""

import time
import re

import os
from pathlib import Path

import pytest
from langchain_core.documents import Document

from services.rag_service import RAGService
from services.vector_store import VectorStoreService
from services.embedding_service import EmbeddingService

TEST_CP_FILE = "tests/documents/CPA52_13-17_ParentsGuide_WEB.txt"

# helper function for tests
def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[*_#`]+', '', text) # remove markdown bold/italics
    text = re.sub(r'\s+', ' ', text) # collapse multiple spaces/newlines
    return text.strip()

# main entry point for RAG pipeline
@pytest.fixture(scope="module")
def rag_service():
    """Fixture to initialize the RAG service."""
    return RAGService()

# helper function: low-level interaction with vector database
@pytest.fixture(scope="module")
def vector_store_service(rag_service) -> VectorStoreService:
    """Shortcut to the inner VectorStoreService."""
    return rag_service.vector_store_service


def test_doc_ingested(
    rag_service: RAGService, vector_store_service: VectorStoreService
) -> RAGService:
    """
    RAG service pre-loaded with the CP article so every test
    in this module re-uses the same temporary vector-DB.
    """
    # start with a clean slate
    vector_store_service.clear()
    stats_before = rag_service.vector_store_service.get_collection_stats()

    if stats_before["total_documents"] == 0:
    
        print("DEBUG: Ingesting article...")

        # ingest the article
        start = time.time()
        rag_service.setup_vector_db(TEST_CP_FILE)
        end = time.time()
        
        print(f"Vectore DB setup took {end - start:.2f} seconds")
        # overall: embedding model makes a HUGE difference: stick to something small for CPU
        # using batches to insert, changing batch size for embedding, and chunk sizes makes minimal differences (1-2s)
    
    stats_after = rag_service.vector_store_service.get_collection_stats()

    assert stats_before["total_documents"] == 0  # tests if VDB was cleared
    assert stats_after["total_documents"] > 0  # tests if the article was ingestd
    assert stats_after["total_documents"] >= 50  # tests if the article was chunked


@pytest.mark.parametrize(
    ("query", "expected_snippet"),
    [
        (
            """What two therapies improve upper limb function 
            for teenagers with CP affecting one side of their body?""",
            "Constraint Induced Movement Therapy and Bimanual Therapy",
        ),
        (
            "Explain why physical activity is important for teens with CP",
            "Physical activity is essential for your teen’s health.",
        ),
        (
            """What is it called when people with CP communicate using 
            methods other than speech or in addition to speech""",
            "Augmentative and Alternative Communication (AAC)",
        ),
        (
            "What is whole body vibration (WBV)?",
            "WBV involves standing, sitting or lying down on a machine with a vibrating platform",
        ),
    ],
)

def test_direct_similarity_search(
    vector_store_service: VectorStoreService, query: str, expected_snippet: str
):

    """Debug: print top retrieved chunks"""
    # query the vector DB directly
    docs = vector_store_service.similarity_search(query, k=5) # old: docs = vector_store_service.similarity_search(query, k=1)

    print("--- DEBUG: Retrieved chunks ---")
    for i, d in enumerate(docs, start=1):
        preview = d.page_content.strip().replace("\n", " ")
        print(f"[{i}] {preview[:300]}{'...' if len(preview) > 300 else ''}")
    
    """Top-1 chunk returned by similarity_search should mention the snippet."""
    assert len(docs) > 0 # old: assert len(docs) == 1
    top_chunk = normalize(docs[0].page_content)
    expected_snippet = normalize(expected_snippet)

    print(f"TOP-1: {top_chunk}")
    print(f"EXPECTED: {expected_snippet}")

    if expected_snippet not in top_chunk:
        found_in_lower = False
        i = 1
        for d in docs[1:]:
            i = i+1
            if expected_snippet in normalize(d.page_content):
                found_in_lower = True
                break
        
        if found_in_lower:
            msg = (f"Expected snippet not in top-1, but in top-{i}\n"
                   f"Snippet: {expected_snippet}"
            )
        else:
            msg = (f"Expected snippet not found in any top-5 retrieved chunks\n"
                   f"Snippet: {expected_snippet}"
            )

        assert False, msg

    assert expected_snippet in top_chunk

@pytest.mark.parametrize(
    ("question", "keywords"),
    [
        (
            "What two therapies improve upper limb function for teenagers with CP affecting one side of their body?",
            ["constraint", "bimanual", "therapy"],
        ),
        (
            "Explain why physical activity is important for teens with CP.",
            ["physical", "activity", "health"],
        ),
        (
            "What is it called when people with CP communicate using methods other than speech or in addition to speech?",
            ["augmentative", "alternative", "communication"],
        ),
        (
            "What is whole body vibration (WBV)?",
            ["WBV", "vibrating", "body"],
        ),
    ],
)

def test_rag_chain_answers(
    rag_service: RAGService, question: str, keywords: list[str]
):
    """Debug: print retrieved content and LLM answer"""
    # create retriever
    retriever = rag_service.vector_store_service.get_retriever(search_type="similarity", k=5)

    # chunks as context passed to LLM
    retrieved_docs = retriever.invoke(question)
    print(f"\n=== DEBUG: Question ===\n{question}\n")
    print("--- Retrieved context chunks ---")
    for i, d in enumerate(retrieved_docs, start=1):
        preview = d.page_content.strip().replace("\n", " ")
        print(f"[{i}] {preview[:300]}{'...' if len(preview) > 300 else ''}")
    
    """Full RAG chain should return an answer that references expected terms."""
    start = time.time()
    rag_chain = rag_service.create_rag_chain()
    
    print(f"RAG setup took {time.time() - start:.2f} seconds")
    
    start = time.time()
    response = rag_chain.invoke(question)
    answer = response if isinstance(response, str) else response.content

    print(f"getting LLM response took {time.time() - start:.2f} seconds")

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