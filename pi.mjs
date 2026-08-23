// pi.mjs — 最简演示：4号模型 + pi 默认工具集（bash/read/edit/write 等），工作目录 specs/
// 运行：node pi.mjs "筏板的最小厚度"
import { readFileSync } from "node:fs";

for (const line of readFileSync(new URL("./.env", import.meta.url), "utf8").split("\n")) {
  const m = line.match(/^\s*([\w.-]+)\s*=\s*(.*)?\s*$/);
  if (m && !(m[1] in process.env)) process.env[m[1]] = (m[2] ?? "").replace(/^["']|["']$/g, "");
}

import { Agent } from "@earendil-works/pi-agent-core";
import { createModels, createProvider, envApiKeyAuth } from "@earendil-works/pi-ai";
import * as openAICompletionsApi from "@earendil-works/pi-ai/api/openai-completions";
import { createReadOnlyTools } from "@earendil-works/pi-coding-agent";

const BASE_URL = "https://opencode.ai/zen/go/v1";
const MODEL_ID = "deepseek-v4-flash";

const models = createModels();
models.setProvider(createProvider({
  id: "cfg4", name: MODEL_ID, baseUrl: BASE_URL,
  auth: { apiKey: envApiKeyAuth("OPENCODE", ["OPENCODE_API_KEY"]) },
  api: openAICompletionsApi,
  models: [{
    id: MODEL_ID, name: MODEL_ID, provider: "cfg4", api: "openai-completions", baseUrl: BASE_URL,
    reasoning: true, input: ["text"], cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
    contextWindow: 131072, maxTokens: 65536,
    compat: { supportsDeveloperRole: false },
  }],
}));

const agent = new Agent({
  initialState: {
    systemPrompt: "你是工程规范检索专家。./specs 目录下是规范文本，先用工具搜索阅读相关条文再回答，回答中注明文件名和行号。",
    model: models.getModel("cfg4", MODEL_ID),
    tools: createReadOnlyTools("./specs"),
    thinkingLevel: "off",
  },
  streamFn: models.streamSimple.bind(models),
});
agent.subscribe(e => {
  if (e.type === "message_update" && e.assistantMessageEvent.type === "text_delta") process.stdout.write(e.assistantMessageEvent.delta);
  else if (e.type === "tool_execution_start") console.log(`\n🔧 ${e.toolName} ${JSON.stringify(e.args)}`);
});

console.log(`🚀 pi Agent | ${MODEL_ID}`);
await agent.prompt(process.argv.slice(2).join(" ") || "筏板的最小厚度");
const lastMsg = agent.state.messages.at(-1);
console.log(`\n✅ 完成 (stopReason=${lastMsg?.stopReason}, blocks=[${lastMsg?.content.map(b => b.type)}]${lastMsg?.errorMessage ? `, 错误=${lastMsg.errorMessage}` : ""})`);
