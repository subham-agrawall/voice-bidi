"""Nova Sonic session management for AWS Bedrock bidirectional streaming."""

import os
import base64
import json
import logging
from typing import Optional
from uuid import uuid4

from google.genai import types

logger = logging.getLogger(__name__)


class NovaSonicSession:
    """Manages a Nova Sonic bidirectional streaming session with AWS Bedrock.

    Handles session lifecycle, event streaming, and audio I/O configuration
    for real-time voice interactions with the Nova-2-Sonic model.
    """

    def __init__(
        self,
        model_id: str,
        config: Optional[types.LiveConnectConfig] = None,
    ):
        """Initialize a Nova Sonic session.

        Args:
            model_id: AWS Bedrock model identifier (e.g., 'amazon.nova-2-sonic-v1:0')
            config: Live connection configuration from google adk
            region: AWS region for Bedrock runtime endpoint
        """

        self.model_id = model_id
        self.region = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
        self.client = None
        self.stream = None
        self.response_task = None
        self.is_active = False
        self.prompt_name = str(uuid4())
        self.content_name = str(uuid4())
        self.audio_content_name = str(uuid4())
        self.role: Optional[str] = None
        self.display_assistant_text: bool = False
        self.config = config
        self.tools = None

        # TBD: should be an input from adk LiveConnectConfig
        self.turn_detection_config = {"endpointingSensitivity": "LOW"}

        # TBD: read from self.config.speech_config.prebuilt_voice_config.voice_name
        self.voice_id = "matthew"

    def _get_config_param(self, param_name: str):
        """Get a parameter from config with fallback hierarchy.
        
        Priority order:
        1. config.param_name (e.g., config.max_output_tokens)
        2. config.generation_config.param_name (e.g., config.generation_config.max_tokens)
        
        Args:
            param_name: The parameter name to retrieve
            
        Returns:
            The parameter value or None if not found
        """
        # Try direct config attribute
        value = getattr(self.config, param_name, None)
        if value is not None:
            return value
        
        # Try generation_config attribute
        if hasattr(self.config, 'generation_config') and self.config.generation_config:
            value = getattr(self.config.generation_config, param_name, None)
        
        return value

    def _initialize_client(self):
        """Initialize the Bedrock runtime client lazily (keeps import-time light).
        """
        from aws_sdk_bedrock_runtime.client import BedrockRuntimeClient
        from aws_sdk_bedrock_runtime.config import Config
        from smithy_aws_core.identity.environment import EnvironmentCredentialsResolver

        config = Config(
            endpoint_uri=f"https://bedrock-runtime.{self.region}.amazonaws.com",
            region=self.region,
            aws_credentials_identity_resolver=EnvironmentCredentialsResolver(),
        )
        self.client = BedrockRuntimeClient(config=config)

    async def send_event(self, event_json: str):
        from aws_sdk_bedrock_runtime.models import InvokeModelWithBidirectionalStreamInputChunk, BidirectionalInputPayloadPart
        # Skip INFO-level logging for raw audio input events to reduce noise.
        if '"audioInput"' in event_json:
            logger.debug("Skipping INFO log for outgoing audioInput event")
        else:
            logger.info("Outgoing event: %s", event_json)

        payload_bytes = event_json.encode("utf-8")
        logger.debug("Outgoing event bytes: %d", len(payload_bytes))

        event = InvokeModelWithBidirectionalStreamInputChunk(
            value=BidirectionalInputPayloadPart(bytes_=payload_bytes)
        )
        await self.stream.input_stream.send(event)

    async def start_session(self):
        if not self.client:
            self._initialize_client()

        from aws_sdk_bedrock_runtime.client import InvokeModelWithBidirectionalStreamOperationInput

        self.stream = await self.client.invoke_model_with_bidirectional_stream(
            InvokeModelWithBidirectionalStreamOperationInput(model_id=self.model_id)
        )
        self.is_active = True

        # Inference configuration
        max_tokens = self._get_config_param("max_output_tokens")
        top_p = self._get_config_param("top_p")
        temperature = self._get_config_param("temperature")

        inference_config = {
            "maxTokens": max_tokens if max_tokens is not None else 1024,
            "topP": top_p if top_p is not None else 0.9,
            "temperature": temperature if temperature is not None else 0.7,
        }

        # Send session start event
        session_start = json.dumps({
            "event": {
                "sessionStart": {
                    "inferenceConfiguration": inference_config,
                    "turnDetectionConfiguration": self.turn_detection_config,
                }
            }
        })
        await self.send_event(session_start)

        # Create json for prompt start event
        prompt_body = {
            "promptName": self.prompt_name,
        }
        
        response_modalities = self._get_config_param("response_modalities")
        
        # Text output configuration — always include
        # text transcriptions (input/output) even when response is audio-only
        prompt_body["textOutputConfiguration"] = {"mediaType": "text/plain"}

        # Audio output configuration if AUDIO requested
        if "AUDIO" in response_modalities:
            audio_config = {
                "mediaType": "audio/lpcm",
                "sampleRateHertz": 24000,
                "sampleSizeBits": 16,
                "channelCount": 1,
                "voiceId": self.voice_id,
                "encoding": "base64",
                "audioType": "SPEECH",
            }
            prompt_body["audioOutputConfiguration"] = audio_config

        # Tool configuration
        tool_config = []
        if hasattr(self.config, "tools") and self.config.tools:
            for tool in self.config.tools:
                
                # Hardcoded for FunctionTool only for now
                # TBD: handle other tool types
                tool = tool.function_declarations[0]
                
                tool_json = {
                    "toolSpec": {
                        "name": tool.name,
                        "description": tool.description,
                        "inputSchema": {
                            "json": tool.parameters.model_dump_json(exclude_none=True),
                        },
                    }
                }
                tool_config.append(tool_json)

        if tool_config:
            prompt_body["toolUseOutputConfiguration"] = {"mediaType": "application/json"}
            prompt_body["toolConfiguration"] = {"tools": tool_config}
        
        # Send prompt start event
        prompt_start = json.dumps({"event": {"promptStart": prompt_body}})
        logger.debug("promptStart event: %s", prompt_start)
        await self.send_event(prompt_start)

        # Send content start event for system message
        text_content_start = json.dumps({
            "event": {
                "contentStart": {
                    "promptName": self.prompt_name,
                    "contentName": self.content_name,
                    "type": "TEXT",
                    "interactive": False,
                    "role": "SYSTEM",
                    "textInputConfiguration": {"mediaType": "text/plain"},
                }
            }
        })
        await self.send_event(text_content_start)

        # Send system instruction as text input event
        text_input = json.dumps({
            "event": {
                "textInput": {
                    "promptName": self.prompt_name,
                    "contentName": self.content_name,
                    "content": self.config.system_instruction,
                }
            }
        })
        await self.send_event(text_input)

        text_content_end = json.dumps({
            "event": {
                "contentEnd": {
                    "promptName": self.prompt_name,
                    "contentName": self.content_name,
                }
            }
        })
        await self.send_event(text_content_end)

    async def start_audio_input(self):
        audio_content = {
            "event": {
                "contentStart": {
                    "promptName": self.prompt_name,
                    "contentName": self.audio_content_name,
                    "type": "AUDIO",
                    "interactive": True,
                    "role": "USER",
                    "audioInputConfiguration": {
                        "mediaType": "audio/lpcm",
                        "sampleRateHertz": 16000,
                        "sampleSizeBits": 16,
                        "channelCount": 1,
                        "audioType": "SPEECH",
                        "encoding": "base64",
                    },
                }
            }
        }

        await self.send_event(json.dumps(audio_content))

    async def send_audio_chunk(self, audio_bytes: bytes):
        if not self.is_active:
            return
        audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
        audio_event = json.dumps({
            "event": {
                "audioInput": {
                    "promptName": self.prompt_name,
                    "contentName": self.audio_content_name,
                    "content": audio_b64,
                }
            }
        })
        await self.send_event(audio_event)

    async def end_audio_input(self):
        audio_content_end = json.dumps({
            "event": {
                "contentEnd": {
                    "promptName": self.prompt_name,
                    "contentName": self.audio_content_name,
                }
            }
        })
        self.audio_content_name = str(uuid4())  # generate new content name for next audio input
        await self.send_event(audio_content_end)

    async def start_text_input(self, content_name: str, role: str):
        """Start a text input content stream.
        
        Args:
            content_name: Content name for this text stream.
        """
        text_content_start = {
            "event": {
                "contentStart": {
                    "promptName": self.prompt_name,
                    "contentName": content_name,
                    "type": "TEXT",
                    "interactive": True,
                    "role": role,
                    "textInputConfiguration": {"mediaType": "text/plain"},
                }
            }
        }
        await self.send_event(json.dumps(text_content_start))

    async def send_text_input(self, text_content: str, content_name: str):
        """Send text input to the model.
        
        Args:
            text_content: The text message to send to the model.
        """
        if not self.is_active:
            return
        
        text_event = json.dumps({
            "event": {
                "textInput": {
                    "promptName": self.prompt_name,
                    "contentName": content_name,
                    "content": text_content,
                }
            }
        })
        await self.send_event(text_event)

    async def end_text_and_tool_input(self, content_name: str):
        """End the text input content stream."""
        text_content_end = json.dumps({
            "event": {
                "contentEnd": {
                    "promptName": self.prompt_name,
                    "contentName": content_name,
                }
            }
        })
        await self.send_event(text_content_end)

    async def start_tool_input(self, content_name: str, tool_use_id: str):
        """Start a tool input content stream.
        
        Args:
            content_name: Content name for this tool stream.
            tool_use_id: The ID of the tool use.
        """
        tool_content_start = {
            "event": {
                "contentStart": {
                    "promptName": self.prompt_name,
                    "contentName": content_name,
                    "interactive": False,
                    "type": "TOOL",
                    "role": "TOOL",
                    "toolResultInputConfiguration": {
                        "toolUseId": tool_use_id,
                        "type": "TEXT",
                        "textInputConfiguration": {
                            "mediaType": "text/plain"
                        }
                    }
                }
            }
        }
        await self.send_event(json.dumps(tool_content_start))

    async def send_tool_input(self, tool_result: str, content_name: str):
        """Send tool input to the model.
        
        Args:
            tool_result: The tool result content to send to the model.
        """
        if not self.is_active:
            return
        
        tool_event = json.dumps({
            "event": {
                "toolResult": {
                    "promptName": self.prompt_name,
                    "contentName": content_name,
                    "content": tool_result,
                }
            }
        })
        await self.send_event(tool_event)

    async def end_session(self):
        if not self.is_active:
            return
        prompt_end = json.dumps({"event": {"promptEnd": {"promptName": self.prompt_name}}})
        await self.send_event(prompt_end)

        session_end = json.dumps({"event": {"sessionEnd": {}}})
        await self.send_event(session_end)
        await self.stream.input_stream.close()
        self.is_active = False
