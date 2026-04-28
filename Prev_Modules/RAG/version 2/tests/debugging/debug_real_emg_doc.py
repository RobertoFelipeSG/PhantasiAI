
# %%
from services.rag_service import RAGService
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from operator import itemgetter
# %%
rag_service = RAGService()
rag_service.vector_store_service.clear()
rag_service.setup_vector_db("tests/documents/Ib--ez_et_al-2025-Nature_Biomedical_Engineering.txt")
print(rag_service.vector_store_service.get_collection_stats()["total_documents"])
rag_chain = rag_service.create_rag_chain()
query = "What are considered high-frequency inputs to muscles in Hz?"
retriever = rag_service.vector_store_service.get_retriever(
    search_type="similarity",
    k=3
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

    {"context": itemgetter("question") | retriever | format_docs, 
     "question": itemgetter("question") }
    | rag_service.llm_service.get_prompt()
    | rag_service.llm_service.get_llm()
    | StrOutputParser()
)


response = chain.invoke({"question": query})
print("response:", response)


# %%
rag_service = RAGService()
# rag_service.vector_store_service.clear()
# rag_service.setup_vector_db("tests/documents/Ib--ez_et_al-2025-Nature_Biomedical_Engineering.txt")
print(rag_service.vector_store_service.get_collection_stats()["total_documents"])
rag_chain = rag_service.create_rag_chain()
query = "On what exact calendar date was the Perspective accepted for publication?"
retriever = rag_service.vector_store_service.get_retriever(
    search_type="similarity",
    k=3
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

    {"context": itemgetter("question") | retriever | format_docs, 
     "question": itemgetter("question") }
    | rag_service.llm_service.get_prompt()
    | rag_service.llm_service.get_llm()
    | StrOutputParser()
)


response = chain.invoke({"question": query})
print("response:", response)
# %%
rag_service = RAGService()
# rag_service.vector_store_service.clear()
# rag_service.setup_vector_db("tests/documents/Ib--ez_et_al-2025-Nature_Biomedical_Engineering.txt")
print(rag_service.vector_store_service.get_collection_stats()["total_documents"])
rag_chain = rag_service.create_rag_chain()
query = "On what exact calendar date was the Perspective accepted for publication?"
retriever = rag_service.vector_store_service.get_retriever(
    search_type="similarity",
    k=3
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

    {"context": itemgetter("question") | retriever | format_docs, 
     "question": itemgetter("question") }
    | rag_service.llm_service.get_prompt()
    | rag_service.llm_service.get_llm()
    | StrOutputParser()
)


response = chain.invoke({"question": query})
print("response:", response)
# %%
rag_service = RAGService()
# rag_service.vector_store_service.clear()
# rag_service.setup_vector_db("tests/documents/Ib--ez_et_al-2025-Nature_Biomedical_Engineering.txt")
print(rag_service.vector_store_service.get_collection_stats()["total_documents"])
rag_chain = rag_service.create_rag_chain()
query = "How many motor neurons were simulated in the population model illustrated in Figure 3?"
retriever = rag_service.vector_store_service.get_retriever(
    search_type="similarity",
    k=3
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

    {"context": itemgetter("question") | retriever | format_docs, 
     "question": itemgetter("question") }
    | rag_service.llm_service.get_prompt()
    | rag_service.llm_service.get_llm()
    | StrOutputParser()
)


response = chain.invoke({"question": query})
print("response:", response)
# %%
rag_service = RAGService()
# rag_service.vector_store_service.clear()
# rag_service.setup_vector_db("tests/documents/Ib--ez_et_al-2025-Nature_Biomedical_Engineering.txt")
print(rag_service.vector_store_service.get_collection_stats()["total_documents"])
rag_chain = rag_service.create_rag_chain()
query = "Which specific lower-limb muscle was instrumented with high-density EMG electrodes in the simultaneous brain–muscle experiment shown in Figure 5a?"
retriever = rag_service.vector_store_service.get_retriever(
    search_type="similarity",
    k=3
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

    {"context": itemgetter("question") | retriever | format_docs, 
     "question": itemgetter("question") }
    | rag_service.llm_service.get_prompt()
    | rag_service.llm_service.get_llm()
    | StrOutputParser()
)


response = chain.invoke({"question": query})
print("response:", response)
# %%
rag_service = RAGService()
# rag_service.vector_store_service.clear()
# rag_service.setup_vector_db("tests/documents/Ib--ez_et_al-2025-Nature_Biomedical_Engineering.txt")
print(rag_service.vector_store_service.get_collection_stats()["total_documents"])
rag_chain = rag_service.create_rag_chain()
query = "What average time lag (in milliseconds) between cortical activity and motor-neuron activity is reported in Figure 5b?"
retriever = rag_service.vector_store_service.get_retriever(
    search_type="similarity",
    k=3
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

    {"context": itemgetter("question") | retriever | format_docs, 
     "question": itemgetter("question") }
    | rag_service.llm_service.get_prompt()
    | rag_service.llm_service.get_llm()
    | StrOutputParser()
)


response = chain.invoke({"question": query})
print("response:", response)
# %%
