# %%
from services.rag_service import RAGService
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from operator import itemgetter

# %%
rag_service = RAGService()
rag_service.vector_store_service.clear()
rag_service.setup_vector_db(
    "tests/documents/tSCS_upper_body.txt"
)
print(rag_service.vector_store_service.get_collection_stats()["total_documents"])
rag_chain = rag_service.create_rag_chain()
query = """
    You are an expert document parser.The document you are reading contains 
    information on transcutaneous spinal cord stimulation (tSCS) extracted from 
    PDF files. Your goal is to extract **only** the following structured 
    attributes, and return the result as a **list of JSON objects**, each 
    corresponding to one row in the table.

    Required fields (per row):

    - "filename" (e.g., "tSCS_upper_body.pdf")
    - "publication" (e.g., "Inanici et al, 2021")
    - "electrode_placement_active" (e.g., "C3/C4 and C6/C7")
    - "electrode_placement_passive" (e.g., "ASIS")
    - "amplitude_mA" (e.g., "40–90 mA")
    - "duration_min_per_session" (e.g., "60 +/- 20")
    - "frequency_Hz_burst" (e.g., "30", or "5Hz–30", or "0.2")

    ### Instructions:
    1. If any field is missing, output it as `null`.
    2. Normalize whitespace and line breaks (e.g., multiline entries should be 
    joined into a single line).
    3. Keep values **verbatim** as they appear in the source.
    4. The final output must be valid JSON only — no commentary, no preamble.

    Output format example:

    ```json
    [
    {
        "filename": "tSCS_lower_body.pdf",
        "publication": null,
        "electrode_placement_active": "T11, T12",
        "electrode_placement_passive": "lower abdomen",
        "amplitude_mA": "90% of the PRM-reflex threshold",
        "duration_min_per_session": "30",
        "frequency_Hz_burst": "50"
    },
    ...
    ]"""

retriever = rag_service.vector_store_service.get_retriever(
    search_type="similarity", k=3
)
docs = retriever.invoke(query)
print("\n--- Retrieved chunks ---")
for d in docs:
    print("---")
    print(d.page_content)


def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)


print(format_docs(docs))

chain = (
    {
        "context": itemgetter("question") | retriever | format_docs,
        "question": itemgetter("question"),
    }
    | rag_service.llm_service.get_prompt()
    | rag_service.llm_service.get_llm()
    | StrOutputParser()
)


response = chain.invoke({"question": query})
print("response:", response)

# %%
