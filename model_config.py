"""模型配置模块 - 统一管理所有LLM模型配置"""

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
        "model_name": "Qwen/Qwen3-235B-A22B-Instruct-2507",
        "thinking": "qwen",
    },
    4: {
        "base_url": "https://api-inference.modelscope.cn/v1",
        "api_key": "modelscope_key",
        "model_name": "Qwen/Qwen3.5-27B",
        "thinking": "qwen",
    },
    5: {
        "base_url": "https://api-inference.modelscope.cn/v1",
        "api_key": "modelscope_key",
        "model_name": "Qwen/Qwen3-30B-A3B-Instruct-2507",
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
        "base_url": "http://127.0.0.1:8013/v1",
        "api_key": "GEMINI_API_KEY",
        "model_name": "gemini-3.1-pro",
        "thinking": "kimi",
    },
    12: {
        "base_url": "https://apihub.agnes-ai.com/v1",
        "api_key": "AGNES_API_KEY",
        "model_name": "agnes-2.0-flash",
        "thinking": "kimi",
    },
    13: {
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "api_key": "ZHIPU_API_KEY",
        "model_name": "glm-4.7-flash",
        "thinking": "kimi",
    },
}
