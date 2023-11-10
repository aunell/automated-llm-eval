import logging
import warnings
from dataclasses import dataclass
from typing import NamedTuple

import httpx
import openai

import private_key

chat_logger = logging.getLogger(name="ChatLogger")


class MessageBundle(NamedTuple):
    "Input messages & API call settings bundled with response messages and metadata."
    id: str
    system_message: str
    user_message: str
    response_message: str
    created_time: int
    model: str
    total_tokens: int
    prompt_tokens: int
    completion_tokens: int
    # Additional API Call Options
    seed: int
    temperature: float
    top_p: float
    max_tokens: int | None


@dataclass(kw_only=True)
class ChatModel:
    """Wrapper around openai.ChatCompletion with concurrency limiting
    and exponential backoff retries."""

    # OpenAI API Config
    sync_client: openai.OpenAI = openai.OpenAI(
        api_key=private_key.key["open-ai"],
        max_retries=10,
        timeout=httpx.Timeout(60.0, read=5.0, write=10.0, connect=10.0),
    )
    async_client: openai.AsyncOpenAI = openai.AsyncOpenAI(
        api_key=private_key.key["open-ai"],
        max_retries=10,
        timeout=httpx.Timeout(60.0, read=5.0, write=10.0, connect=10.0),
    )
    # Model Config
    model: str = "gpt-3.5-turbo"
    temperature: float = 0.9
    top_p: float = 0.9
    max_tokens: int = None
    n: int = 1
    default_batch_size: int = 1
    seed: int = 42

    def create_chat_completion(
        self, system_message: str, user_message: str, output_format: str | None = "simple", **kwargs
    ) -> openai.types.chat.chat_completion.ChatCompletion | str | MessageBundle | dict:
        """Simplified Chat Completion call for `n=1` and parses output.

        Args:
            system_message (str): system message prompt
            user_message (str): user message prompt
            output_format (str | None, optional): Controls format of output.
                `None`: return raw ChatCompletion object
                `simple`: return only response message
                `message_bundle`: return namedtuple with input+output messages and ChatCompletion
                    metadata flattened as a namedtuple
                `message_bundle_dict`: same as `message_bundle`, but returns as a
                    dictionary.

        Returns:
            Either ChatCompletion, string response message, MessageBundle, or dict depending
            on `output_format`.
        """
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ]
        cc = self.chat_completion(messages=messages, **kwargs)
        match output_format:
            case "simple":
                chat_completion_message = cc.choices[0].message.content
                return chat_completion_message
            case "message_bundle" | "message_bundle_dict":
                api_call_kwargs = {
                    "seed": self.seed,
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "max_tokens": self.max_tokens,
                } | kwargs
                mb = MessageBundle(
                    id=cc.id,
                    system_message=system_message,
                    user_message=user_message,
                    response_message=cc.choices[0].message.content,
                    created_time=cc.created,
                    model=cc.model,
                    total_tokens=cc.usage.total_tokens,
                    prompt_tokens=cc.usage.prompt_tokens,
                    completion_tokens=cc.usage.completion_tokens,
                    **api_call_kwargs,
                )
                if output_format == "message_bundle_dict":
                    return mb._asdict()
                else:
                    return mb
            case None:
                return cc

    def chat_completion(
        self, messages: list[dict[str, str]], **kwargs
    ) -> openai.types.chat.chat_completion.ChatCompletion:
        """Calls OpenAI ChatCompletions API.
        https://platform.openai.com/docs/api-reference/chat/create

        This method uses properties declared on class as default arguments.
        Any keyword arguments directly passed in to `kwargs` will override
        the default arguments.

        Args:
            messages (list[dict[str, str]]): _description_

        Returns:
            openai.types.chat.chat_completion.ChatCompletion: _description_
        """
        default_kwargs = {
            "messages": messages,
            "model": self.model,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "n": self.n,
            "seed": self.seed,
        }
        updated_kwargs = default_kwargs | kwargs
        try:
            return self.sync_client.chat.completions.create(**updated_kwargs)
        except Exception:
            warnings.warn(f"Failed to create ChatCompletion with arguments: {kwargs}")
            return None
