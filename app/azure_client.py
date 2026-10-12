import json
import logging
import re
from typing import Any, Dict, Optional, Tuple

from openai import AzureOpenAI, OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from app.config import settings

logger = logging.getLogger(__name__)

# GPT-4o pricing estimates per 1k tokens for cost tracking
PROMPT_COST_PER_1K = 0.005
COMPLETION_COST_PER_1K = 0.015


class LLMClient:
    def __init__(self):
        self.client = None
        self.deployment = settings.AZURE_OPENAI_DEPLOYMENT
        self.is_azure = False
        self._init_client()

    def _init_client(self):
        if settings.AZURE_OPENAI_API_KEY and settings.AZURE_OPENAI_ENDPOINT:
            try:
                self.client = AzureOpenAI(
                    api_key=settings.AZURE_OPENAI_API_KEY,
                    api_version=settings.AZURE_OPENAI_API_VERSION,
                    azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
                    timeout=30.0,
                    max_retries=2,
                )
                self.is_azure = True
                self.deployment = settings.AZURE_OPENAI_DEPLOYMENT
                logger.info(f"Initialized Azure OpenAI client with deployment: {self.deployment}")
            except Exception as e:
                logger.warning(f"Failed to initialize Azure OpenAI client: {e}")
        elif settings.OPENAI_API_KEY:
            try:
                self.client = OpenAI(
                    api_key=settings.OPENAI_API_KEY,
                    timeout=30.0,
                    max_retries=2,
                )
                self.deployment = settings.OPENAI_MODEL
                self.is_azure = False
                logger.info(f"Initialized standard OpenAI client with model: {self.deployment}")
            except Exception as e:
                logger.warning(f"Failed to initialize OpenAI client: {e}")

    @property
    def is_configured(self) -> bool:
        return self.client is not None and not settings.MOCK_LLM

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True
    )
    def _execute_api_call(self, system_prompt: str, user_prompt: str) -> Tuple[str, Dict[str, int]]:
        """Executes API call to Azure OpenAI or OpenAI with temperature 0."""
        response = self.client.chat.completions.create(
            model=self.deployment,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content or "{}"
        usage = {
            "prompt": response.usage.prompt_tokens if response.usage else 0,
            "completion": response.usage.completion_tokens if response.usage else 0,
            "total": response.usage.total_tokens if response.usage else 0,
        }
        return content, usage

    def call_llm_json(
        self,
        system_prompt: str,
        user_prompt: str,
        mock_handler: Optional[callable] = None
    ) -> Tuple[Dict[str, Any], Dict[str, int], float]:
        """
        Calls the LLM requiring JSON output.
        Retries once if the first completion is not valid JSON.
        Returns: (parsed_json_dict, token_usage_dict, estimated_cost_usd)
        """
        if not self.is_configured:
            # Deterministic offline mock mode
            if mock_handler:
                parsed_json = mock_handler(system_prompt, user_prompt)
            else:
                parsed_json = {"score": 5.0, "reasoning": "Deterministic offline mock evaluation."}
            prompt_toks = len(user_prompt.split()) + len(system_prompt.split())
            completion_toks = len(str(parsed_json).split())
            usage = {
                "prompt": prompt_toks,
                "completion": completion_toks,
                "total": prompt_toks + completion_toks,
            }
            cost = (prompt_toks / 1000.0 * PROMPT_COST_PER_1K) + (completion_toks / 1000.0 * COMPLETION_COST_PER_1K)
            return parsed_json, usage, round(cost, 6)

        # Real API call with retry once on invalid JSON
        content, usage = self._execute_api_call(system_prompt, user_prompt)
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            # Retry once with explicit json correction prompt
            logger.warning("LLM produced invalid JSON. Retrying once with correction prompt...")
            repair_prompt = (
                f"{user_prompt}\n\nIMPORTANT: Your previous output was not valid JSON:\n{content}\n"
                "Return ONLY a syntactically valid JSON object adhering strictly to the requested schema."
            )
            content, retry_usage = self._execute_api_call(system_prompt, repair_prompt)
            usage["prompt"] += retry_usage["prompt"]
            usage["completion"] += retry_usage["completion"]
            usage["total"] += retry_usage["total"]

            try:
                parsed = json.loads(content)
            except json.JSONDecodeError:
                # Fallback: extract json block using regex
                match = re.search(r"\{.*\}", content, re.DOTALL)
                if match:
                    parsed = json.loads(match.group(0))
                else:
                    raise ValueError(f"Failed to decode JSON from LLM judge: {content}")

        cost = (usage["prompt"] / 1000.0 * PROMPT_COST_PER_1K) + (usage["completion"] / 1000.0 * COMPLETION_COST_PER_1K)
        return parsed, usage, round(cost, 6)


# Singleton client
llm_client = LLMClient()
