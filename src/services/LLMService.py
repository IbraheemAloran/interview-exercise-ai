from typing import Type, TypeVar
from google import genai
from pydantic import BaseModel, ValidationError
from logger_config import get_logger

T = TypeVar("T", bound=BaseModel)


class LLMServiceError(Exception):
    """Base exception for LLM service errors."""


class LLMService:

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-3-flash-preview",
    ):
        self.model = model
        self.logger = get_logger(__name__)

        try:
            self.client = genai.Client(api_key=api_key)
            self.logger.info("Gemini client initialized")

        except Exception as e:
            self.logger.exception("Failed to initialize Gemini client")
            raise LLMServiceError("Client initialization failed") from e

    async def generate(
        self,
        prompt: str,
        response_model: Type[T],
        temperature: float = 0.0,
    ) -> T:
        #Generate structured response from LLM and validate via Pydantic.
        

        try:
            self.logger.info("Sending request to Gemini")

            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config={
                    "response_mime_type": "application/json",
                    "response_json_schema": response_model.model_json_schema(),
                    "temperature": temperature,
                },

            )

            if not response.text:
                raise LLMServiceError("Empty response from model")

            self.logger.debug("Raw LLM response: %s", response.text)

            parsed = response_model.model_validate_json(response.text)

            self.logger.info("Response validated successfully")

            return parsed.model_dump()

        except ValidationError as ve:
            self.logger.exception("Schema validation failed")
            raise LLMServiceError("LLM returned invalid schema") from ve

        except Exception as e:
            self.logger.exception("LLM generation failed")
            raise LLMServiceError("LLM generation error") from e