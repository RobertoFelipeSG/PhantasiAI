from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from core.config_new import settings
from dotenv import load_dotenv
import logging
import torch
import os

load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACE_HUB_TOKEN")

logger = logging.getLogger(__name__)

class LLMService:
    def __init__(self):
        
        self.model_name = settings.LLM_MODEL
        self.device = settings.LLM_DEVICE

        logger.info(f"Using device: {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            use_auth_token=HF_TOKEN
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            use_auth_token=HF_TOKEN,
            device_map={"": "cpu"},
            low_cpu_mem_usage=True,
            #load_in_4bit=True,
            torch_dtype=torch.float32,
        ).to(self.device)
        
        self.prompt = """
        Answer the question based only on the following context:

        Context: {context}

        Question: {question}

        Please provide a clear and concise answer. If the context doesn't contain the information needed, 
        say "I cannot answer this based on the provided context."

        Answer: 
        """

        logger.info(f"Initialized LLM service with model: {settings.LLM_MODEL}")

    def build_prompt(self, context: str, question: str) -> str:
        return self.prompt_template.format(context=context, question=question)

    def generate(self, context: str, question: str, max_new_tokens: int = 256) -> str:
        prompt = self.build_prompt(context, question)
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,        
                use_cache=True          
            )
        
        return self.tokenizer.decode(output[0], skip_special_tokens=True)
    
    def get_llm(self):
        return self.model

    def get_prompt(self):
        return self.prompt
