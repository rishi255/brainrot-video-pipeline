from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_core.language_models.chat_models import BaseChatModel
from dotenv import load_dotenv

load_dotenv()


def get_llm(model_name: str, temperature: float = 0) -> BaseChatModel:
    if model_name == "gpt-4o-mini":
        return ChatOpenAI(model="gpt-4o-mini", temperature=temperature)
    elif model_name == "llama-3.3-70b-versatile":
        return ChatGroq(model="llama-3.3-70b-versatile", temperature=temperature)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
