"""
utils/llm.py
Central LLM loader — swap model name here to change for all agents.
Requires Ollama running locally: `ollama serve` and `ollama pull llama3`
"""

from langchain_ollama import ChatOllama


def get_llm(model: str = "llama3", temperature: float = 0.0) -> ChatOllama:
    """
    Returns a LangChain-compatible Ollama LLM.
    temperature=0.0 for deterministic outputs (best for data analysis tasks).
    """
    return ChatOllama(
        model=model,
        temperature=temperature,
        # Increase context window for long log/report analysis
        num_ctx=4096,
    )
