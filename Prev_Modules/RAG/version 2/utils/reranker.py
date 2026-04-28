from langchain.schema import Document
from sentence_transformers import CrossEncoder

class RerankingRetriever:
    def __init__(self, base_retriever, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.base_retriever = base_retriever
        self.reranker = CrossEncoder(model_name)

    def get_relevant_documents(self, query: str):
        # score each (query, doc) pair
        docs = self.base_retriever.get_relevant_documents(query)

        # ensure Document objects
        flat_docs = []
        for d in docs:
            if isinstance(d, list):
                flat_docs.extend(d)      # flatten if a list
            else:
                flat_docs.append(d)

        pairs = [(query, d.page_content) for d in docs]

        # sort (descending)
        scores = self.reranker.predict(pairs)
        reranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
        return [[d for score, d in reranked]]

