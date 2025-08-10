from core.logging import setup_logging
from core.config import settings
from services.rag_service import RAGService
import os


class RAGApplication:
    def __init__(self):
        setup_logging()
        self.rag_service = RAGService()

    def initialize_database(self):
        """Initialize the vector database if needed."""
        if not os.path.exists(settings.DB_DIR):
            sample_path = os.path.join(settings.DATA_DIR, "sample.txt")
            if not os.path.exists(sample_path):
                raise FileNotFoundError(
                    f"Error: '{sample_path}' not found. Please create this file with sample content."
                )
            self.rag_service.setup_vector_db(sample_path)

    def run_interactive(self):
        """Run the interactive query loop."""
        print("\nRAG System initialized! Type 'exit' to quit.")
        while True:
            question = input("\nEnter your question: ")
            if question.lower() == 'exit':
                break

            print("\nGenerating response...")
            response = self.rag_service.query(question)
            print(f"\nResponse: {response}")

    def query(self, question: str) -> str:
        """Single query interface for the application."""
        return self.rag_service.query(question)
