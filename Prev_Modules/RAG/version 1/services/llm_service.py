from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from ..core.config import settings
import logging

logger = logging.getLogger(__name__)

class LLMService:
    def __init__(self):
        self.model = ChatOllama(model=settings.LLM_MODEL)
        self.prompt = ChatPromptTemplate.from_template("""
        Answer the question based only on the following context:

        Context: {context}

        Question: {question}

        Please provide a clear and concise answer. If the context doesn't contain the information needed, 
        say "I cannot answer this based on the provided context."

        Answer: """)

        logger.info(f"Initialized LLM service with model: {settings.LLM_MODEL}")

    def get_llm(self):
        return self.model

    def get_prompt(self):
        return self.prompt
