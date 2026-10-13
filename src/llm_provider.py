from __future__ import annotations

import json
import logging
from typing import Any, Dict

from openai import OpenAI

from config.settings import (
    LLM_MODEL,
    LLM_PROVIDER,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
)

LOGGER = logging.getLogger(__name__)


class LLMProvider:
    def __init__(self):
        self.provider = LLM_PROVIDER
        self.model_name = LLM_MODEL
        self.api_key = OPENAI_API_KEY
        self.base_url = OPENAI_BASE_URL
        self._client: OpenAI | None = None
        LOGGER.info(
            'Initialized LLM provider=%s model=%s base_url=%s configured=%s',
            self.provider,
            self.model_name,
            self.base_url or '<unset>',
            self.is_configured(),
        )

    def is_configured(self) -> bool:
        return bool(self.provider) and (
            (self.provider == 'openai_compatible' and bool(self.api_key and self.base_url))
            or (self.provider == 'ollama' and bool(OLLAMA_BASE_URL))
        )

    def generate(self, prompt: str, system_prompt: str | None = None) -> Dict[str, Any]:
        if self.provider == 'ollama':
            return self._call_ollama(prompt, system_prompt)
        if self.provider != 'openai_compatible':
            raise ValueError(f'Unsupported LLM_PROVIDER: {self.provider}')
        return self._call_openai_compatible(prompt, system_prompt)

    def _call_openai_compatible(self, prompt: str, system_prompt: str | None = None) -> Dict[str, Any]:
        if not self.api_key:
            raise ValueError('OpenAI-compatible model is not configured. Set OPENAI_API_KEY in .env.')
        if not self.base_url:
            raise ValueError('OpenAI-compatible model is not configured. Set OPENAI_BASE_URL in .env.')
        if self._client is None:
            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        response = self._client.chat.completions.create(
            model=self.model_name,
            messages=[
                {'role': 'system', 'content': system_prompt or 'Answer using the provided context only.'},
                {'role': 'user', 'content': prompt},
            ],
            temperature=0.1,
        )
        text = response.choices[0].message.content or ''
        return {'text': text}

    def _call_ollama(self, prompt: str, system_prompt: str | None = None) -> Dict[str, Any]:
        import requests

        payload = {
            'model': OLLAMA_MODEL or 'llama3.1',
            'prompt': prompt,
            'system': system_prompt or 'Answer using the provided context only.',
            'stream': False,
        }
        response = requests.post(f'{OLLAMA_BASE_URL.rstrip("/")}/api/generate', json=payload, timeout=180)
        response.raise_for_status()
        data = response.json()
        return {'text': data.get('response', '')}


def parse_json_response(raw_text: str) -> Dict[str, Any]:
    if not raw_text:
        return {'answer': 'No answer returned.', 'references': [], 'insufficient_evidence': True, 'warning': 'Empty response from LLM.'}
    text = raw_text.strip()
    if text.startswith('```'):
        text = text.strip('`')
        if text.lower().startswith('json'):
            text = text[4:].strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {'answer': raw_text, 'references': [], 'insufficient_evidence': True, 'warning': 'LLM returned non-JSON output and the answer could not be validated.'}
