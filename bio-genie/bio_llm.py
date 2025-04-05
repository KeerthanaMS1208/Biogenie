from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_huggingface import HuggingFaceEndpoint
import os
from dotenv import load_dotenv

load_dotenv()
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

prompt = PromptTemplate(
    input_variables=["question"],
    template="Answer this biology question in a short and simple way:\n\nQuestion: {question}\n\nAnswer:"
)

llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.1",
    huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
    temperature=0.5,
    max_new_tokens=256
)

qa_chain = LLMChain(prompt=prompt, llm=llm)

def ask_bio_question(question: str) -> str:
    return qa_chain.run(question)
