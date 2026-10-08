"""模型配置模块 - 统一管理所有LLM模型配置"""

import uuid

_SESSION_CACHE = {}


def build_default_headers(config):
    """生成供应商所需的默认请求头。

    OpenCode Go 要求每次会话携带稳定的 x-opencode-session，
    并要求客户端用自定义 User-Agent 标识自己。
    """
    headers = {"User-Agent": "llm-rg-search/1.0"}
    base_url = config.get("base_url", "")
    if "opencode.ai" in base_url:
        headers["x-opencode-session"] = _SESSION_CACHE.setdefault(
            base_url, uuid.uuid4().hex
        )
    return headers


MODEL_CONFIG = {
    1: {
        "base_url": "https://api.moonshot.cn/v1",
        "api_key": "kimi_key",
        "model_name": "kimi-k2.5",
        "thinking": "kimi",
    },
    2: {
        "base_url": "https://integrate.api.nvidia.com/v1",
        "api_key": "nvidia_key",
        "model_name": "z-ai/glm-5.2",
    },
    3: {
        "base_url": "https://api-inference.modelscope.cn/v1",
        "api_key": "modelscope_key",
        "model_name": "Qwen/Qwen3.5-122B-A10B",
        "thinking": "qwen",
    },
    4: {
        "base_url": "https://api-inference.modelscope.cn/v1",
        "api_key": "modelscope_key",
        "model_name": "Qwen/Qwen3.8-27B",
        "thinking": "qwen",
    },
    5: {
        "base_url": "https://api-inference.modelscope.cn/v1",
        "api_key": "modelscope_key",
        "model_name": "Qwen/Qwen3.5-35B-A3B",
        "thinking": "qwen",
    },
    6: {
        "base_url": "https://ollama.com/v1",
        "api_key": "ollama_key",
        "model_name": "gemma4:31b-cloud",
    },
    7: {
        "base_url": "https://integrate.api.nvidia.com/v1",
        "api_key": "nvidia_key",
        "model_name": "qwen/qwen3.5-397b-a17b",
    },
    8: {
        "base_url": "https://api.deepseek.com/v1",
        "api_key": "deepseek_key",
        "model_name": "deepseek-v4-flash",
        "thinking": "deepseek",
    },
    9: {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "api_key": "DASHSCOPE_API_KEY",
        "model_name": "qwen3.7-plus",
        "thinking": "qwen",
    },
    10: {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "api_key": "DASHSCOPE_API_KEY",
        "model_name": "qwen3.6-35b-a3b",
        "thinking": "qwen",
    },
    11: {
        "base_url": "http://127.0.0.1:4000/v1",
        "api_key": "GEMINI_API_KEY",
        "model_name": "gemini-3.5-flash-lite",
        "thinking": "kimi",
    },
    12: {
        "base_url": "https://apihub.agnes-ai.com/v1",
        "api_key": "AGNES_API_KEY",
        "model_name": "agnes-2.0-flash",
        "thinking": "kimi",
    },
    13: {
        "base_url": "https://opencode.ai/zen/go/v1",
        "api_key": "OPENCODE_API_KEY",
        "model_name": "deepseek-v4.1-flash",
        "thinking": "deepseek",
    },
}
