from langchain_core.language_models.chat_models import BaseChatModel
from app.core.config import Settings


class LLMFactory:
    def __init__(self, settings: Settings):
        self.settings = settings

    def create_chat_model(self) -> BaseChatModel:
        provider = (self.settings.llm_provider or "").lower().strip()

        if provider == "groq":
            if not self.settings.groq_api_key:
                raise ValueError("Missing GROQ_API_KEY in environment.")
            from langchain_groq import ChatGroq

            return ChatGroq(
                api_key=self.settings.groq_api_key,
                model=self.settings.groq_model,
                temperature=self.settings.llm_temperature,
                max_tokens=self.settings.llm_max_tokens,
            )

        if provider == "google":
            if not self.settings.google_api_key:
                raise ValueError("Missing GOOGLE_API_KEY in environment.")
            from langchain_google_genai import ChatGoogleGenerativeAI

            return ChatGoogleGenerativeAI(
                google_api_key=self.settings.google_api_key,
                model=self.settings.google_model,
                temperature=self.settings.llm_temperature,
                max_output_tokens=self.settings.llm_max_tokens,
            )

        raise ValueError(f"Unsupported LLM_PROVIDER: {self.settings.llm_provider}")
