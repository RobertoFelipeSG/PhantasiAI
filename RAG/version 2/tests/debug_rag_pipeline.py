
# %%
from services.rag_service import RAGService
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from operator import itemgetter
# %%
rag_service = RAGService()
rag_service.vector_store_service.clear()
rag_service.setup_vector_db("tests/documents/fake_client_info.txt")
print(rag_service.vector_store_service.get_collection_stats()["total_documents"])
rag_chain = rag_service.create_rag_chain()
query = "What is the client phone number?"
retriever = rag_service.vector_store_service.get_retriever(
    search_type="similarity",
    k=3
)
docs = retriever.invoke(query)
print("\n--- Retrieved chunks ---")
for d in docs:
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
print(response)


# %%