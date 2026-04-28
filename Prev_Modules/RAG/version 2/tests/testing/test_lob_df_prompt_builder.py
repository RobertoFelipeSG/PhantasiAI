# %%
import re

from pathlib import Path
import json
from typing import List

from services.rag_service import RAGService
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter

# new helpers
from services.prompt_builder.factory import Factory
from services.output_formatter.factory import PromptOutputFormatter

# your Pydantic schema
from services.document_models.study_row_LB_DF import LB_DF_StudyRow, LB_DF_MultipleStudies
# %%
builder = Factory(LB_DF_StudyRow)
prompt_text = builder.assemble_prompt()
json_skel = PromptOutputFormatter(LB_DF_MultipleStudies).generate_json_structure()
full_prompt = (
    f"{prompt_text}\n\nBelow is the exact JSON schema you MUST fill:\n{json_skel}"
)
# %%
rag_service = RAGService()
rag_service.vector_store_service.clear()
rag_service.setup_vector_db(
    ["tests/documents/Hope & Field-Fote, 2023.txt", "tests/documents/Meyer et al., 2020.txt"]
)

retriever = rag_service.vector_store_service.get_retriever(
    search_type="similarity", k=500
)


def format_docs(docs):
    return "\n\n".join(p.page_content for p in docs)


# LangChain runnable chain
chain = (
    {
        "context": itemgetter("question") | retriever | format_docs,
        "question": itemgetter("question"),
    }
    | rag_service.llm_service.get_prompt()
    | rag_service.llm_service.get_llm()
    | StrOutputParser()
)
# %%
# (optional) sanity-print the chunks the retriever fed to the LLM
''' print("\n--- Retrieved chunks ---")
for d in retriever.invoke(full_prompt):
    print("•", d.page_content[:120].replace("\n", " "), "…")
'''

# fire the LLM
raw_text = chain.invoke({"question": full_prompt})

print("\nRAW LLM OUTPUT\n--------------")
print(raw_text)


# %%
try:
    # remove markdown code fences
    clean_text = re.sub(r"```(?:json)?\s*", "", raw_text)
    clean_text = re.sub(r"```", "", clean_text)

    data = json.loads(clean_text)
    if not isinstance(data, list):
        # tolerate the model returning a single object
        data = [data]
except json.JSONDecodeError as e:
    raise ValueError("LLM response was not valid JSON!") from e

rows: List[LB_DF_StudyRow] = []
for obj in data:
    try:
        rows.append(LB_DF_StudyRow.model_validate(obj))
    except Exception as e:
        print("⚠️  Validation error on object:", obj)
        raise

print("\nHYDRATED PYDANTIC OBJECT(S)\n---------------------------")
for row in rows:
    print(row.model_dump_json(indent=2))

# %%
