from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Optional

from google.genai import types
from google.adk.models.base_llm import BaseLlm

from .novasonic_session import NovaSonicSession
from .novasonic_llm_connection import NovaSonicLlmConnection

logger = logging.getLogger(__name__)


class NovaSonic(BaseLlm):
    """Nova Sonic model—drop-in replacement for Gemini."""

    model: str = "amazon.nova-2-sonic-v1:0"
    speech_config: Optional[types.SpeechConfig] = None

    @classmethod
    def supported_models(cls) -> list[str]:
        return [r"amazon.nova-.*"]
    
    @asynccontextmanager
    async def connect(self, llm_request):
        """Connect to Nova Sonic and yield a connection object.

        The implementation mirrors the Gemini.connect() control flow; only
        difference is that we start an embedded Nova Sonic session instance.
        """
        # TBD:
        # 1. some http options set are ignored for nova sonic - seems not needed
        # 2. session_resumption settings are ignored for now..will have to check how it works in nova sonic

        # Same as google_llm.py
        if self.speech_config is not None:
            llm_request.live_connect_config.speech_config = self.speech_config
    
        llm_request.live_connect_config.system_instruction = llm_request.config.system_instruction
        llm_request.live_connect_config.tools = llm_request.config.tools

        logger.info('Trying to connect to live model: %s', llm_request.model)
        logger.debug('Connecting to live with llm_request:%s', llm_request)
        logger.debug('Live connect config: %s', llm_request.live_connect_config)

        ns = NovaSonicSession(
            model_id=llm_request.model,
            config=llm_request.live_connect_config,
        )
        ns._initialize_client()
        await ns.start_session()
        try:
            yield NovaSonicLlmConnection(ns)
        finally:
            await ns.end_session()

    async def generate_content_async(self, llm_request, stream: bool = False):
        """Not implemented for Nova Sonic live adapter.

        Nova Sonic uses bidirectional live sessions via `connect()`; synchronous
        or SSE-style unidirectional generation is not supported by this adapter.
        This method exists to satisfy the BaseLlm abstract interface and will
        raise an explicit error guiding callers to use `connect()` instead.
        """
        raise NotImplementedError(
            'generate_content_async is not supported for live Nova Sonic sessions; call `connect()` instead.'
        )