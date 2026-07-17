"""DeepSeek V4 Pro 大模型封装。

使用官方 ``langchain-deepseek`` 集成包，模型固定为 ``deepseek-v4-pro``。
API Key 通过环境变量 ``DEEPSEEK_API_KEY`` 注入。
"""

import json
from functools import lru_cache
from typing import Any, Iterator, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.outputs.chat_generation import ChatGenerationChunk

from greenhouse_agent import config


@lru_cache(maxsize=1)
def get_llm() -> BaseChatModel:
    """返回全局共享的 DeepSeek V4 Pro Chat 模型实例。

    使用 ``lru_cache`` 保证整个进程只创建一次模型客户端。
    """
    if not config.DEEPSEEK_API_KEY:
        raise RuntimeError(
            "未配置 DEEPSEEK_API_KEY 环境变量，请在 .env 文件中设置。"
        )

    # 延迟导入，避免在没有 DeepSeek 依赖时也能 import 本模块的其他部分
    from langchain_deepseek import ChatDeepSeek

    return ChatDeepSeek(
        model=config.DEEPSEEK_MODEL,
        api_key=config.DEEPSEEK_API_KEY,
        base_url=config.DEEPSEEK_BASE_URL,
        temperature=config.DEEPSEEK_TEMPERATURE,
        max_tokens=config.DEEPSEEK_MAX_TOKENS,
        timeout=60,
        max_retries=2,
    )


def invoke_llm(prompt: str, system: Optional[str] = None) -> str:
    """同步调用 DeepSeek V4 Pro 生成回答。

    Args:
        prompt: 用户输入 prompt。
        system: 可选的系统提示词，用于约束模型角色与输出风格。

    Returns:
        模型生成的文本内容。
    """
    from langchain_core.messages import HumanMessage, SystemMessage

    llm = get_llm()
    messages: list[BaseMessage] = []
    if system:
        messages.append(SystemMessage(content=system))
    messages.append(HumanMessage(content=prompt))

    response = llm.invoke(messages)
    return response.content if hasattr(response, "content") else str(response)


def stream_llm(prompt: str, system: Optional[str] = None) -> Iterator[str]:
    """流式调用 DeepSeek V4 Pro，逐段返回生成内容。

    用于在 Streamlit 中通过 ``st.write_stream`` 实时显示生成结果。
    """
    from langchain_core.messages import HumanMessage, SystemMessage

    llm = get_llm()
    messages: list[BaseMessage] = []
    if system:
        messages.append(SystemMessage(content=system))
    messages.append(HumanMessage(content=prompt))

    for chunk in llm.stream(messages):
        # type: BaseChatModelChunk / ChatGenerationChunk
        if isinstance(chunk, ChatGenerationChunk):
            text = chunk.message.content
        else:
            text = getattr(chunk, "content", "")  # type: ignore[arg-type]
        if text:
            yield text


def invoke_llm_json(prompt: str, system: Optional[str] = None) -> dict:
    """调用 DeepSeek V4 Pro 强制输出 JSON，并解析为 dict。

    使用 OpenAI 兼容的 ``response_format={"type": "json_object"}`` 约束输出。
    若模型未返回有效 JSON，抛出 ``ValueError``。

    Args:
        prompt: 用户输入 prompt（应明确要求 JSON 输出）
        system: 可选的系统提示词

    Returns:
        解析后的 dict
    """
    from langchain_core.messages import HumanMessage, SystemMessage

    llm = get_llm()
    # 强制 JSON 输出模式（DeepSeek 兼容 OpenAI 接口）
    bound = llm.bind(response_format={"type": "json_object"})
    messages: list[BaseMessage] = []
    if system:
        messages.append(SystemMessage(content=system))
    messages.append(HumanMessage(content=prompt))

    response = bound.invoke(messages)
    content = response.content if hasattr(response, "content") else str(response)
    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        raise ValueError(f"DeepSeek 未返回有效 JSON：{e}\n原文：{content[:500]}")
