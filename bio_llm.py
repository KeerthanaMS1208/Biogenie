import os
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint
from langchain.chains import LLMChain

load_dotenv()

HUGGINGFACEHUB_API_TOKEN = os.getenv("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")

llm = HuggingFaceEndpoint(
    repo_id="google/flan-t5-base",
    huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
    model_kwargs={
        "temperature": 0.5,
        "max_new_tokens": 128  # ✅ under 250
    }
)

prompt = PromptTemplate(
    input_variables=["question"],
    template="You are a biology expert. Answer the following question in simple terms:\n\nQ: {question}\nA:"
)

qa_chain = LLMChain(prompt=prompt, llm=llm)

def ask_bio_question(question: str) -> str:
    try:
        return qa_chain.run(question)
    except Exception as e:
        return {"error": str(e)}
