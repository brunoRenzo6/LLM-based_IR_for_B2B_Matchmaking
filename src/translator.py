"""
translator.py - LLM-based translation from Brazilian Portuguese to English.

Translation is applied to:
  - Project descriptions (queries) before display
  - Seller snippet summaries after summarization, before markdown formatting

Enable / disable via config.TRANSLATE_TO_ENGLISH.
"""

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import AzureChatOpenAI

import config
from io_utils import read_json

parser = StrOutputParser()


def translate_to_english(model: AzureChatOpenAI, text: str) -> str:
    """
    Translate *text* from Brazilian Portuguese to English using the LLM.

    Returns the original text unchanged when:
      - config.TRANSLATE_TO_ENGLISH is False
      - the input is empty / whitespace-only
    """
    if not config.TRANSLATE_TO_ENGLISH:
        return text

    if not text or not text.strip():
        return text

    prompts = read_json("prompts/prompts_translate.json")
    messages = [
        SystemMessage(content=prompts["m1"]),
        HumanMessage(content=text),
    ]
    return parser.invoke(model.invoke(messages))
