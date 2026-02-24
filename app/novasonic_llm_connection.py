"""Nova Sonic LLM connection adapter for Google ADK.

Implements BaseLlmConnection interface to enable Nova Sonic model integration
with the ADK's bidirectional streaming architecture.
"""

import base64
import json
import logging
from typing import AsyncGenerator
from typing import Union
from uuid import uuid4

from google.adk.utils.content_utils import filter_audio_parts
from google.genai import types
from google.adk.models.base_llm_connection import BaseLlmConnection
from google.adk.models.llm_response import LlmResponse

from .novasonic_session import NovaSonicSession

logger = logging.getLogger(__name__)

RealtimeInput = Union[types.Blob, types.ActivityStart, types.ActivityEnd]


class NovaSonicLlmConnection(BaseLlmConnection):
    """Thin wrapper around NovaSonicSession to implement BaseLlmConnection.

    Translates ADK `send_*` and `receive` semantics into Nova Sonic events
    using the session's `send_event` / audio helpers. Mirrors Gemini's
    connection semantics so behavior remains consistent.
    """

    def __init__(self, ns_client: NovaSonicSession):
        self._ns = ns_client
        self._input_transcription_text: str = ''
        self._output_transcription_text: str = ''
        self._current_completion_id: str | None = None
        self._audio_input_content_open: bool = False
        
        # Map ADK roles → Nova Sonic roles (for sending)
        self.role_mapping = {
            'user': 'USER',
            'model': 'ASSISTANT',
            'system': 'SYSTEM',
        }
        # Map Nova Sonic roles → ADK roles (for receiving)
        self._inbound_role_mapping = {
            'USER': 'user',
            'ASSISTANT': 'model',
            'SYSTEM': 'system',
        }

    def _is_control_text(self, text: str) -> bool:
        """Filter out Nova Sonic control signals disguised as text.
        
        Nova Sonic sometimes sends control signals as JSON text that should
        not be displayed in the UI.
        """
        if not text or not text.strip():
            return True
            
        try:
            parsed = json.loads(text.strip())
            # Filter out known control signals
            control_keys = ['interrupted', 'turn_complete', 'event']
            return any(key in parsed for key in control_keys)
        except json.JSONDecodeError:
            return False

    async def send_history(self, history: list[types.Content]):
        """Sends the conversation history to the Nova Sonic model.

        You call this method right after setting up the model connection.
        The model will respond if the last content is from user; otherwise, it will
        wait for new user input before responding.

        Args:
          history: The conversation history to send to the model.
        """

        # Filter out audio parts from history
        contents = [
            filtered
            for content in history
            if (filtered := filter_audio_parts(content)) is not None
        ]

        # Send each content from history as text events
        if contents:
            logger.debug('Sending history to live connection: %s', contents)
            for content in contents:
                role = self.role_mapping.get(content.role)
                text = ''.join(part.text for part in content.parts)
                if not text:
                    continue
                
                # Send chat history
                content_name = str(uuid4())
                await self._ns.start_text_input(content_name=content_name, role=role)
                await self._ns.send_text_input(text, content_name=content_name)
                await self._ns.end_text_input(content_name=content_name)
                logger.debug('Sent history content with role %s (Nova: %s): %s', content.role, role, text)
        else:
            logger.info('no content is sent')

    async def send_content(self, content: types.Content):
        """Sends a user content to the Nova Sonic model.

        The model will respond immediately upon receiving the content.

        Args:
          content: The content to send to the model.
        """
        assert content.parts
        logger.debug('Sending LLM new content %s', content)

        content_name = str(uuid4())
        role = self.role_mapping.get(content.role)
        text = ''.join(part.text for part in content.parts)
        await self._ns.start_text_input(content_name=content_name, role=role)
        await self._ns.send_text_input(text, content_name=content_name)
        await self._ns.end_text_input(content_name=content_name)

    async def send_realtime(self, input: RealtimeInput):
        """Sends a chunk of audio or activity signal to the model in realtime.

        Supports both server-side VAD (just blobs, no activity signals) and
        client-side VAD (ActivityStart → blobs → ActivityEnd).

        For server-side VAD: lazily opens an audio content block on the first
        blob and keeps it open for the session lifetime.

        For client-side VAD: ActivityStart opens a new audio content block,
        ActivityEnd closes it.

        Args:
        input: The input to send to the model.
        """

        if isinstance(input, types.Blob):
            # Lazy open: if no audio content block is open, open one now.
            # This handles server-side VAD where no ActivityStart is sent.
            if not self._audio_input_content_open:
                logger.debug('Auto-opening audio input content block (server-side VAD).')
                await self._ns.start_audio_input()
                self._audio_input_content_open = True
            logger.debug('Sending LLM Blob.')
            await self._ns.send_audio_chunk(input.data)

        elif isinstance(input, types.ActivityStart):
            # Client-side VAD: explicitly open audio content block
            if not self._audio_input_content_open:
                logger.debug('Sending LLM activity start signal.')
                await self._ns.start_audio_input()
                self._audio_input_content_open = True
            else:
                logger.debug('Activity start received but audio input content block already open, ignoring.')

        elif isinstance(input, types.ActivityEnd):
            # Client-side VAD: explicitly close audio content block
            if self._audio_input_content_open:
                logger.debug('Sending LLM activity end signal.')
                await self._ns.end_audio_input()
                self._audio_input_content_open = False
            else:
                logger.debug('Activity end received but no audio input content block open, ignoring.')

        else:
            raise ValueError('Unsupported input type: %s' % type(input))

    async def receive(self) -> AsyncGenerator[LlmResponse, None]:
        """Receives the model response from the Nova Sonic stream.

        Yields:
          LlmResponse: The model response.
        """
        self._active_contents: dict[str, str] = {}

        while self._ns.is_active:
            # Await next message from Nova Sonic stream
            output = await self._ns.stream.await_output()
            result = await output[1].receive()
            
            if not (result.value and result.value.bytes_):
                continue

            # Response event
            response_data = result.value.bytes_.decode('utf-8')
            json_data = json.loads(response_data)
            event = json_data.get("event", {})
            logger.debug('Response event from model: %s', event)

            # Handle completionStart — model is about to generate output
            # No ADK yield needed; just reset state for the new completion.
            if "completionStart" in event:
                completion = event["completionStart"]
                self._current_completion_id = completion.get("completionId")
                self._input_transcription_text = ''
                self._output_transcription_text = ''
                continue

            # Handle usage events
            if "usageEvent" in event:
                usage = event["usageEvent"]
                usage_metadata = types.GenerateContentResponseUsageMetadata(
                    prompt_token_count=usage.get("totalInputTokens"),
                    candidates_token_count=usage.get("totalOutputTokens"),
                    total_token_count=usage.get("totalTokens"),
                )
                continue
                # yield LlmResponse(
                #     usage_metadata=usage_metadata,
                # )
                # continue

            # contentStart events
            if "contentStart" in event:
                event_cs = event["contentStart"]
                content_id = event_cs["contentId"]
                content_type = event_cs.get("type") # "TEXT" or "AUDIO"
                content_role = self._inbound_role_mapping.get(event_cs.get("role"))
                
                # Check for completion id
                completion_id = event_cs.get("completionId")
                if completion_id!=self._current_completion_id:
                    raise(ValueError(f"Received contentStart for completionId {completion_id} which doesn't match current completion id {self._current_completion_id}"))

                # filter SPECULATIVE content for TEXT type
                if content_type=="TEXT":
                    additional_fields = event_cs.get("additionalModelFields")
                    if isinstance(additional_fields, str):
                        additional_fields = json.loads(additional_fields)
                    generative_stage = additional_fields.get("generationStage")
                    if generative_stage == "SPECULATIVE":
                        logger.debug('Skipping content for speculative generation stage with content id: %s', content_id)
                        continue
                
                # Active contents
                self._active_contents[content_id] = content_role

            
            # Handle textOutput — route to input or output transcription
            if "textOutput" in event:
                content_id = event["textOutput"].get("contentId")
                if content_id not in self._active_contents:
                    continue
                    
                text_chunk = event["textOutput"].get("content")
                content_role = self._active_contents[content_id]

                # Skip control signals disguised as text
                if self._is_control_text(text_chunk):
                    logger.debug('Skipping control text: %s', text_chunk)
                    continue

                # TEXT/USER = input transcription (STT of what user said)
                if content_role == "user":
                    self._input_transcription_text += text_chunk
                    yield LlmResponse(
                        input_transcription=types.Transcription(
                            text=text_chunk,
                            finished=False,
                        ),
                        partial=True,
                    )
                    continue

                # TEXT/ASSISTANT = output transcription (text of what model is speaking)
                if content_role == "model":
                    self._output_transcription_text += text_chunk
                    yield LlmResponse(
                        output_transcription=types.Transcription(
                            text=text_chunk,
                            finished=False,
                        ),
                        partial=True,
                    )
                    continue

            # Handle audioOutput event — yield audio as blob
            if "audioOutput" in event:
                content_id = event["audioOutput"].get("contentId")
                audio_b64 = event["audioOutput"].get("content")
                if audio_b64:
                    try:
                        audio_bytes = base64.b64decode(audio_b64)
                    except Exception:
                        logger.debug("Failed to decode audio payload")
                        continue
                    blob = types.Blob(data=audio_bytes, mime_type="audio/pcm")
                    content = types.Content(
                        role=self._active_contents[content_id],
                        parts=[types.Part(inline_data=blob)],
                    )
                    yield LlmResponse(content=content)
                continue

            # Handle contentEnd — flush transcription buffers based on stopReason
            if "contentEnd" in event:
                event_ce = event["contentEnd"]
                content_id = event_ce.get("contentId")
                if content_id not in self._active_contents:
                    continue
                    
                stop_reason = event_ce.get("stopReason")
                content_type = event_ce.get("type")
                content_role = self._active_contents.get(content_id)
                logger.debug(
                    'Content end: contentId=%s, type=%s, role=%s, stopReason=%s',
                    content_id, content_type, content_role, stop_reason,
                )

                
                if content_type == "TEXT":
                    # Flush input transcription for USER text
                    if self._input_transcription_text:
                        yield LlmResponse(
                            input_transcription=types.Transcription(
                                text=self._input_transcription_text,
                                finished=True,
                            ),
                            partial=False,
                        )
                        self._input_transcription_text = ''
                        yield LlmResponse(turn_complete=True)
                        continue


                    # Flush output transcription for ASSISTANT text only at
                    if self._output_transcription_text:
                        yield LlmResponse(
                            output_transcription=types.Transcription(
                                text=self._output_transcription_text,
                            finished=True,
                            ),
                            partial=False,
                        )
                        self._output_transcription_text = ''

                if stop_reason == "END_TURN":
                    yield LlmResponse(turn_complete=True)
                elif stop_reason == "INTERRUPTED":
                    yield LlmResponse(interrupted=True)

                # Remove from active contents tracking
                self._active_contents.pop(content_id, None)
                continue

            # Handle completionEnd
            if "completionEnd" in event:
                # Check if there are no active contents
                if self._active_contents:
                    raise(RuntimeError(f"Received completionEnd event but there are still active contents: {self._active_contents}"))
                
                yield LlmResponse(turn_complete=True)
                break


    async def close(self):
        """Closes the Nova Sonic connection."""
        try:
            # Close any open audio content block before ending session
            if self._audio_input_content_open:
                logger.debug('Closing audio input content block before session end.')
                await self._ns.end_audio_input()
                self._audio_input_content_open = False
            await self._ns.end_session()
        except Exception as e:
            logger.debug("Error closing Nova Sonic session: %s", e)
