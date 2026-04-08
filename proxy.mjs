import http from "node:http";
import crypto from "node:crypto";
import { execSync } from "node:child_process";

process.on("uncaughtException", (err) => {
  console.error("[proxy] uncaught exception:", err.message);
});
process.on("unhandledRejection", (err) => {
  console.error("[proxy] unhandled rejection:", err.message || err);
});

const PORT = process.env.PROXY_PORT || 4000;

const MINIMAX_BASE = process.env.MINIMAX_BASE_URL || "https://api.minimax.io/v1";
const MINIMAX_KEY = process.env.MINIMAX_API_KEY || "";
const MINIMAX_MODELS = parseCsv(process.env.MINIMAX_MODELS || "MiniMax-M2.7");

const OPENAI_BASE = process.env.OPENAI_BASE_URL || "https://api.openai.com/v1";
const OPENAI_KEY = process.env.OPENAI_API_KEY || "";
const OPENAI_MODELS = parseCsv(process.env.OPENAI_MODELS || "gpt-5.4,gpt-5.4-mini,gpt-5.4-nano,gpt-4o");
const OPENAI_MODEL_PREFIXES = parseCsv(process.env.OPENAI_MODEL_PREFIXES || "gpt-,o1,o3,o4,codex-,chatgpt-");

const OPENROUTER_KEY = process.env.OPENROUTER_API_KEY || "";
const OPENROUTER_BASE = "https://openrouter.ai/api/v1";
const OPENROUTER_SEARCH_MODEL = process.env.OPENROUTER_SEARCH_MODEL || "nvidia/nemotron-3-super-120b-a12b:free";

const DEFAULT_PROVIDER = (process.env.DEFAULT_PROVIDER || "").trim().toLowerCase();
const GITHUB_TOKEN = process.env.GITHUB_TOKEN || (() => {
  try { return execSync("gh auth token", { encoding: "utf-8", timeout: 3000 }).trim(); }
  catch { return ""; }
})();

if (!MINIMAX_KEY && !OPENAI_KEY) {
  console.error("At least one upstream provider key is required: set MINIMAX_API_KEY and/or OPENAI_API_KEY");
  process.exit(1);
}
if (!OPENROUTER_KEY) {
  console.warn("[proxy] OPENROUTER_API_KEY not set — MiniMax web_search requests will be dropped");
}

const enabledProviders = new Set();
if (MINIMAX_KEY) enabledProviders.add("minimax");
if (OPENAI_KEY) enabledProviders.add("openai");

const providerModels = {
  minimax: MINIMAX_MODELS,
  openai: OPENAI_MODELS,
};

const explicitModelProvider = new Map();
for (const model of MINIMAX_MODELS) explicitModelProvider.set(normalizeModelId(model), "minimax");
for (const model of OPENAI_MODELS) explicitModelProvider.set(normalizeModelId(model), "openai");

const modelCatalog = [
  ...MINIMAX_MODELS.map((id) => ({ id, object: "model", owned_by: "minimax" })),
  ...OPENAI_MODELS.map((id) => ({ id, object: "model", owned_by: "openai" })),
];

// --- Response store for previous_response_id bridging ---

const responseStore = new Map();
const STORE_TTL = 60 * 60 * 1000; // 1 hour
const STORE_MAX = 500;
const MAX_CONSECUTIVE_TOOL_CALLS = 20; // circuit breaker threshold

// --- Proxy-side web_fetch tool (bypasses sandbox restrictions) ---

const WEB_FETCH_TOOL = {
  type: "function",
  function: {
    name: "web_fetch",
    description: "Fetch content from a URL over HTTP/HTTPS. Use this when you need to retrieve content from a web URL. Returns HTTP status and response body as clean markdown (via Jina Reader for HTML pages). Supports all HTTP methods.",
    parameters: {
      type: "object",
      properties: {
        url: { type: "string", description: "The URL to fetch (http:// or https://)" },
        method: { type: "string", enum: ["GET", "HEAD", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"], description: "HTTP method (default: GET)" },
        headers: { type: "object", description: "Optional HTTP headers as key-value pairs" },
        body: { type: "string", description: "Request body for POST/PUT/PATCH requests" },
      },
      required: ["url"],
    },
  },
};

// --- Jina Reader integration for clean markdown fetches ---

const JINA_BASE = "https://r.jina.ai";
const JINA_FETCH_TIMEOUT = 20000;
const JINA_MAX_BODY = 80000;

async function jinaRead(url) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), JINA_FETCH_TIMEOUT);
  try {
    const res = await fetch(`${JINA_BASE}/${url}`, {
      signal: controller.signal,
      headers: {
        "Accept": "text/plain",
        "X-Return-Format": "markdown",
        "User-Agent": "Mozilla/5.0 (compatible; CodexProxy/1.0)",
      },
    });
    clearTimeout(timeout);
    if (!res.ok) {
      const text = await res.text().catch(() => "");
      return `Jina error: ${res.status} ${res.statusText}\n${text}`.slice(0, JINA_MAX_BODY);
    }
    let text = await res.text();
    if (text.length > JINA_MAX_BODY) {
      text = text.slice(0, JINA_MAX_BODY) + `\n...[content truncated, ${text.length - JINA_MAX_BODY} chars omitted]`;
    }
    return text;
  } catch (err) {
    clearTimeout(timeout);
    if (err.name === "AbortError") return "Jina fetch error: request timed out (20s)";
    return `Jina fetch error: ${err.message}`;
  }
}

const MAX_FETCH_LOOPS = 5;
const FETCH_TIMEOUT = 15000;
const FETCH_MAX_BODY = 50000;

async function rawFetch(url, method = "GET", headers = {}, reqBody = null) {
  if (!headers["User-Agent"]) headers["User-Agent"] = "Mozilla/5.0 (compatible; CodexProxy/1.0)";
  if (GITHUB_TOKEN && /api\.github\.com/.test(url) && !headers["Authorization"] && !headers["authorization"]) {
    headers["Authorization"] = `Bearer ${GITHUB_TOKEN}`;
  }
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), FETCH_TIMEOUT);
  const fetchOpts = { method, headers, signal: controller.signal, redirect: "follow" };
  if (reqBody && /^(POST|PUT|PATCH)$/i.test(method)) fetchOpts.body = reqBody;
  const response = await fetch(url, fetchOpts);
  clearTimeout(timeout);
  const ct = response.headers.get("content-type") || "";
  const status = `HTTP ${response.status} ${response.statusText}`;
  if (/^(HEAD|OPTIONS)$/i.test(method)) {
    const hdrs = [...response.headers.entries()].map(([k, v]) => `${k}: ${v}`).join("\n");
    return `${status}\n${hdrs}`;
  }
  if (/image|audio|video|octet-stream/.test(ct)) {
    return `${status}\nContent-Type: ${ct}\n(binary content, not shown)`;
  }
  let text = await response.text();
  if (text.length > FETCH_MAX_BODY) {
    text = text.slice(0, FETCH_MAX_BODY) + `\n...[truncated, ${text.length - FETCH_MAX_BODY} chars omitted]`;
  }
  return `${status}\n\n${text}`;
}

async function executeWebFetch(argsStr) {
  try {
    const args = typeof argsStr === "string" ? JSON.parse(argsStr) : argsStr;
    const { url, method = "GET", headers = {}, body: reqBody } = args;
    if (!url) return "Error: no URL provided";
    if (method === "GET") return await jinaRead(url);
    return await rawFetch(url, method, headers, reqBody);
  } catch (err) {
    if (err.name === "AbortError") return "Fetch error: request timed out";
    return `Fetch error: ${err.message}`;
  }
}

function parseCsv(value) {
  return [...new Set(
    String(value || "")
      .split(",")
      .map((s) => s.trim())
      .filter(Boolean)
  )];
}

function normalizeModelId(model) {
  return String(model || "").trim().toLowerCase();
}

function contentHasUrl(content) {
  if (typeof content === "string") return /https?:\/\//.test(content);
  if (Array.isArray(content)) {
    return content.some((part) => {
      if (typeof part === "string") return /https?:\/\//.test(part);
      if (part && typeof part.text === "string") return /https?:\/\//.test(part.text);
      if (part && typeof part.url === "string") return /https?:\/\//.test(part.url);
      if (part && typeof part.image_url === "string") return /https?:\/\//.test(part.image_url);
      if (part?.image_url?.url && typeof part.image_url.url === "string") return /https?:\/\//.test(part.image_url.url);
      return false;
    });
  }
  return false;
}

function conversationHasUrls(messages) {
  return messages.some((message) => contentHasUrl(message?.content));
}

function ensureWebFetchTool(tools) {
  const list = Array.isArray(tools) ? [...tools] : [];
  const alreadyPresent = list.some((tool) => {
    if (tool?.type !== "function") return false;
    return tool?.function?.name === WEB_FETCH_TOOL.function.name || tool?.name === WEB_FETCH_TOOL.function.name;
  });
  if (!alreadyPresent) list.push(WEB_FETCH_TOOL);
  return list;
}

function ensureWebFetchHint(messages) {
  const hint =
    "[System: You have a `web_fetch` tool available for making HTTP requests. Use it instead of curl, wget, or other shell-based HTTP tools. Call web_fetch with {\"url\": \"...\"} to fetch any URL. It supports GET, HEAD, POST, PUT, DELETE, PATCH, and OPTIONS methods.]";
  const alreadyPresent = messages.some((message) => message?.role === "user" && message?.content === hint);
  if (alreadyPresent) return messages;
  return [...messages, { role: "user", content: hint }];
}

function getFallbackProvider() {
  if (DEFAULT_PROVIDER && enabledProviders.has(DEFAULT_PROVIDER)) return DEFAULT_PROVIDER;
  if (enabledProviders.has("openai")) return "openai";
  if (enabledProviders.has("minimax")) return "minimax";
  throw new Error("No providers are enabled");
}

function resolveProviderForModel(model) {
  const normalized = normalizeModelId(model);
  if (normalized) {
    const explicit = explicitModelProvider.get(normalized);
    if (explicit && enabledProviders.has(explicit)) return explicit;
    if (enabledProviders.has("minimax") && normalized.includes("minimax")) return "minimax";
    if (enabledProviders.has("openai")) {
      const looksOpenAI = OPENAI_MODEL_PREFIXES.some((prefix) => normalized.startsWith(prefix.toLowerCase()));
      if (looksOpenAI) return "openai";
    }
  }
  return getFallbackProvider();
}

function storeResponse(id, data) {
  if (!id) return;

  if (responseStore.size >= STORE_MAX) {
    const now = Date.now();
    for (const [key, val] of responseStore) {
      if (now - val.storedAt > STORE_TTL) responseStore.delete(key);
    }
    if (responseStore.size >= STORE_MAX) {
      const oldest = responseStore.keys().next().value;
      responseStore.delete(oldest);
    }
  }

  const isToolCallOnly = Array.isArray(data.output) &&
    data.output.length > 0 &&
    data.output.every((o) => o.type === "function_call");

  let consecutiveToolCalls = 0;
  if (isToolCallOnly && data.previousResponseId) {
    const prev = responseStore.get(data.previousResponseId);
    consecutiveToolCalls = (prev?.consecutiveToolCalls || 0) + 1;
  }

  responseStore.set(id, { ...data, storedAt: Date.now(), consecutiveToolCalls });
  console.log(
    `[proxy] stored response ${id} (provider=${data.provider || "unknown"}, store size: ${responseStore.size}${consecutiveToolCalls > 0 ? `, consecutive_tc: ${consecutiveToolCalls}` : ""})`
  );
}

function resolveResponseChain(previousResponseId) {
  const chain = [];
  let currentId = previousResponseId;
  const visited = new Set();

  while (currentId && !visited.has(currentId)) {
    visited.add(currentId);
    const stored = responseStore.get(currentId);
    if (!stored) {
      console.warn(`[proxy] previous_response_id ${currentId} not found in store`);
      break;
    }
    chain.unshift(stored);
    currentId = stored.previousResponseId;
  }

  const items = [];
  for (const entry of chain) {
    if (Array.isArray(entry.input)) items.push(...entry.input);
    if (Array.isArray(entry.output)) items.push(...entry.output);
  }
  return items;
}

function normalizeInputToArray(input) {
  if (Array.isArray(input)) return input;
  if (typeof input === "string") {
    return [{ type: "message", role: "user", content: [{ type: "input_text", text: input }] }];
  }
  return [];
}

function maybeResolvePreviousResponseChain(body, targetProvider) {
  if (!body.previous_response_id) return;

  const previous = responseStore.get(body.previous_response_id);
  if (!previous) {
    if (targetProvider === "minimax") {
      console.warn(`[proxy] previous_response_id ${body.previous_response_id} missing; MiniMax request will continue without restored history`);
    }
    return;
  }

  const needsLocalResolution = targetProvider === "minimax" || previous.provider !== targetProvider;
  if (!needsLocalResolution) return;

  const chainItems = resolveResponseChain(body.previous_response_id);
  if (chainItems.length === 0) return;

  const currentInput = normalizeInputToArray(body.input);
  body.input = [...chainItems, ...currentInput];
  delete body.previous_response_id;
  console.log(`[proxy] locally resolved previous_response_id across provider boundary -> ${targetProvider} (${chainItems.length} items prepended)`);
}

// --- Request translation: Responses API -> Chat Completions (MiniMax path only) ---

function responsesRequestToChatCompletions(body) {
  const messages = [];

  if (body.instructions) {
    messages.push({
      role: "user",
      content: "[System Instructions] " + body.instructions + "\n\nNote: Be efficient with tool calls. Avoid repeating the same tool call unnecessarily.",
    });
  }

  if (typeof body.input === "string") {
    messages.push({ role: "user", content: body.input });
  } else if (Array.isArray(body.input)) {
    let pendingToolCalls = [];

    for (const item of body.input) {
      if (item.type === "message") {
        const role = (item.role === "developer" || item.role === "system") ? "user" : item.role;
        let content;

        if (typeof item.content === "string") {
          content = item.content;
        } else if (Array.isArray(item.content)) {
          content = item.content.map((block) => {
            if (block.type === "input_text") return { type: "text", text: block.text };
            if (block.type === "output_text") return { type: "text", text: block.text };
            if (block.type === "input_image") {
              return { type: "image_url", image_url: { url: block.image_url || block.url } };
            }
            return block;
          });
          if (content.length === 1 && content[0].type === "text") {
            content = content[0].text;
          }
        }

        if (pendingToolCalls.length > 0 && role === "assistant") {
          messages.push({ role: "assistant", content: null, tool_calls: pendingToolCalls });
          pendingToolCalls = [];
        } else {
          if (pendingToolCalls.length > 0) {
            messages.push({ role: "assistant", content: null, tool_calls: pendingToolCalls });
            pendingToolCalls = [];
          }
          messages.push({ role, content });
        }
      } else if (item.type === "function_call") {
        pendingToolCalls.push({
          id: item.call_id || item.id,
          type: "function",
          function: { name: item.name, arguments: item.arguments },
        });
      } else if (item.type === "function_call_output") {
        if (pendingToolCalls.length > 0) {
          messages.push({ role: "assistant", content: null, tool_calls: pendingToolCalls });
          pendingToolCalls = [];
        }
        messages.push({ role: "tool", tool_call_id: item.call_id, content: item.output });
      }
    }

    if (pendingToolCalls.length > 0) {
      messages.push({ role: "assistant", content: null, tool_calls: pendingToolCalls });
    }
  }

  const fixed = [];
  for (let i = 0; i < messages.length; i++) {
    const msg = messages[i];
    if (msg === null) {
      continue;
    } else if (msg.role === "assistant" && msg.tool_calls) {
      fixed.push(msg);
      const callIds = new Set(msg.tool_calls.map((tc) => tc.id));
      for (let j = i + 1; j < messages.length; j++) {
        if (messages[j]?.role === "tool" && callIds.has(messages[j].tool_call_id)) {
          fixed.push(messages[j]);
          messages[j] = null;
        }
      }
    } else if (msg.role === "tool") {
      const lastTc = [...fixed].reverse().find((m) => m.role === "assistant" && m.tool_calls);
      if (lastTc) {
        let insertIdx = fixed.indexOf(lastTc) + 1;
        while (insertIdx < fixed.length && fixed[insertIdx].role === "tool") insertIdx++;
        fixed.splice(insertIdx, 0, msg);
        messages[i] = null;
      }
    } else {
      fixed.push(msg);
    }
  }

  const merged = [];
  for (const msg of fixed) {
    const prev = merged[merged.length - 1];
    if (
      prev &&
      prev.role === msg.role &&
      msg.role === "user" &&
      typeof prev.content === "string" &&
      typeof msg.content === "string"
    ) {
      prev.content += "\n\n" + msg.content;
    } else if (
      prev &&
      prev.role === msg.role &&
      msg.role === "assistant" &&
      !prev.tool_calls &&
      !msg.tool_calls &&
      typeof prev.content === "string" &&
      typeof msg.content === "string"
    ) {
      prev.content += "\n\n" + msg.content;
    } else if (
      prev &&
      prev.role === "assistant" &&
      msg.role === "assistant" &&
      !prev.tool_calls &&
      msg.tool_calls
    ) {
      merged[merged.length - 1] = msg;
    } else if (
      prev &&
      prev.role === "assistant" &&
      msg.role === "assistant" &&
      prev.tool_calls &&
      !msg.tool_calls
    ) {
      // Drop text-only assistant that follows tool calls.
    } else {
      merged.push(msg);
    }
  }

  const TOOL_OUTPUT_MAX = 2000;
  const KEEP_RECENT_FULL = 10;
  for (let i = 0; i < Math.max(0, merged.length - KEEP_RECENT_FULL); i++) {
    const msg = merged[i];
    if (msg.role === "tool" && typeof msg.content === "string" && msg.content.length > TOOL_OUTPUT_MAX) {
      msg.content = msg.content.slice(0, TOOL_OUTPUT_MAX) + "\n...[output truncated, " + (msg.content.length - TOOL_OUTPUT_MAX) + " chars removed]";
    }
  }

  const MAX_MESSAGES = 55;
  let finalMessages = merged;
  if (merged.length > MAX_MESSAGES) {
    const head = merged.slice(0, 2);
    let tail = merged.slice(-(MAX_MESSAGES - 3));
    while (tail.length > 0 && tail[0].role === "tool") tail.shift();
    finalMessages = [
      ...head,
      {
        role: "user",
        content: "[Earlier conversation trimmed. Do not repeat previous statements or tool calls you already made. Continue with the current task. If you have enough information, respond to the user instead of making more tool calls.]",
      },
      ...tail,
    ];
    console.log(`[proxy] trimmed ${merged.length} -> ${finalMessages.length} messages`);
  }

  const validated = [];
  for (const msg of finalMessages) {
    if (msg.role === "tool") {
      const prev = validated[validated.length - 1];
      if (prev && (prev.role === "tool" || (prev.role === "assistant" && prev.tool_calls))) {
        validated.push(msg);
      }
    } else {
      validated.push(msg);
    }
  }
  finalMessages = validated;

  const req = {
    model: body.model,
    messages: finalMessages,
    stream: body.stream || false,
  };

  if (body.temperature != null) req.temperature = body.temperature;
  if (body.top_p != null) req.top_p = body.top_p;
  req.max_tokens = body.max_output_tokens || 16384;

  if (body.tools?.length > 0) {
    const supported = body.tools.filter((t) => t.type === "function");
    if (supported.length > 0) {
      req.tools = supported.map((t) => {
        if (!t.function) {
          return {
            type: "function",
            function: { name: t.name, description: t.description, parameters: t.parameters },
          };
        }
        return t;
      });
    }
  }

  if (body.tool_choice != null) {
    if (typeof body.tool_choice === "object" && body.tool_choice.name) {
      req.tool_choice = { type: "function", function: { name: body.tool_choice.name } };
    } else {
      req.tool_choice = body.tool_choice;
    }
  }

  if (body.reasoning?.effort) req.reasoning_effort = body.reasoning.effort;
  if (body.parallel_tool_calls != null) req.parallel_tool_calls = body.parallel_tool_calls;

  return req;
}

// --- Response translation: Chat Completions -> Responses (MiniMax/OpenRouter path) ---

function uid() {
  return crypto.randomBytes(12).toString("base64url");
}

function chatCompletionToResponse(cc, model, previousResponseId, metadata) {
  const responseId = `resp_${uid()}`;
  const output = [];
  const choice = cc.choices?.[0];

  if (!choice) {
    return {
      id: responseId,
      object: "response",
      created_at: cc.created || Math.floor(Date.now() / 1000),
      status: "completed",
      model: model || cc.model,
      output: [],
      usage: translateUsage(cc.usage),
    };
  }

  const msg = choice.message;

  if (msg.tool_calls?.length > 0) {
    for (const tc of msg.tool_calls) {
      output.push({
        type: "function_call",
        id: `fc_${uid()}`,
        call_id: tc.id,
        name: tc.function.name,
        arguments: tc.function.arguments,
        status: "completed",
      });
    }
  }

  let text = msg.content || "";
  text = text.replace(/<think>[\s\S]*?<\/think>\s*/g, "").trim();
  if (text) {
    output.push({
      type: "message",
      id: `msg_${uid()}`,
      status: "completed",
      role: "assistant",
      content: [{ type: "output_text", text, annotations: [] }],
    });
  }

  if (msg.refusal) {
    const msgItem = output.find((o) => o.type === "message") || {
      type: "message",
      id: `msg_${uid()}`,
      status: "completed",
      role: "assistant",
      content: [],
    };
    msgItem.content.push({ type: "refusal", refusal: msg.refusal });
    if (!output.find((o) => o.type === "message")) output.push(msgItem);
  }

  let status = "completed";
  let incompleteDetails = null;
  if (choice.finish_reason === "length") {
    status = "incomplete";
    incompleteDetails = { reason: "max_output_tokens" };
  } else if (choice.finish_reason === "content_filter") {
    status = "incomplete";
    incompleteDetails = { reason: "content_filter" };
  }

  return {
    id: responseId,
    object: "response",
    created_at: cc.created || Math.floor(Date.now() / 1000),
    status,
    model: model || cc.model,
    output,
    previous_response_id: previousResponseId || null,
    metadata: metadata || {},
    usage: translateUsage(cc.usage),
    incomplete_details: incompleteDetails,
  };
}

function translateUsage(u) {
  if (!u) return { input_tokens: 0, output_tokens: 0, total_tokens: 0 };
  return {
    input_tokens: u.prompt_tokens || 0,
    output_tokens: u.completion_tokens || 0,
    total_tokens: u.total_tokens || 0,
    input_tokens_details: { cached_tokens: u.prompt_tokens_details?.cached_tokens || 0 },
    output_tokens_details: { reasoning_tokens: u.completion_tokens_details?.reasoning_tokens || 0 },
  };
}

// --- Streaming translation for MiniMax chat completions -> Responses SSE ---

function buildStreamingResponseEvents(responseId, model, previousResponseId, metadata) {
  const baseResponse = {
    id: responseId,
    object: "response",
    created_at: Math.floor(Date.now() / 1000),
    status: "in_progress",
    model,
    output: [],
    previous_response_id: previousResponseId || null,
    metadata: metadata || {},
    usage: { input_tokens: 0, output_tokens: 0, total_tokens: 0 },
  };

  return {
    created: () => `event: response.created\ndata: ${JSON.stringify({ type: "response.created", response: baseResponse })}\n\n`,
    inProgress: () => `event: response.in_progress\ndata: ${JSON.stringify({ type: "response.in_progress", response: baseResponse })}\n\n`,
    outputItemAdded: (index, item) => `event: response.output_item.added\ndata: ${JSON.stringify({ type: "response.output_item.added", output_index: index, item })}\n\n`,
    contentPartAdded: (outIdx, contentIdx, part) => `event: response.content_part.added\ndata: ${JSON.stringify({ type: "response.content_part.added", output_index: outIdx, content_index: contentIdx, part })}\n\n`,
    textDelta: (outIdx, contentIdx, delta) => `event: response.output_text.delta\ndata: ${JSON.stringify({ type: "response.output_text.delta", output_index: outIdx, content_index: contentIdx, delta })}\n\n`,
    textDone: (outIdx, contentIdx, text) => `event: response.output_text.done\ndata: ${JSON.stringify({ type: "response.output_text.done", output_index: outIdx, content_index: contentIdx, text })}\n\n`,
    contentPartDone: (outIdx, contentIdx, part) => `event: response.content_part.done\ndata: ${JSON.stringify({ type: "response.content_part.done", output_index: outIdx, content_index: contentIdx, part })}\n\n`,
    outputItemDone: (outIdx, item) => `event: response.output_item.done\ndata: ${JSON.stringify({ type: "response.output_item.done", output_index: outIdx, item })}\n\n`,
    fnCallArgsDelta: (outIdx, callId, delta) => `event: response.function_call_arguments.delta\ndata: ${JSON.stringify({ type: "response.function_call_arguments.delta", output_index: outIdx, call_id: callId, delta })}\n\n`,
    fnCallArgsDone: (outIdx, callId, args) => `event: response.function_call_arguments.done\ndata: ${JSON.stringify({ type: "response.function_call_arguments.done", output_index: outIdx, call_id: callId, arguments: args })}\n\n`,
    completed: (response) => `event: response.completed\ndata: ${JSON.stringify({ type: "response.completed", response })}\n\n`,
  };
}

async function handleStreamingResponse(upstreamRes, res, model, previousResponseId, metadata) {
  res.writeHead(200, {
    "Content-Type": "text/event-stream",
    "Cache-Control": "no-cache",
    Connection: "keep-alive",
  });

  const responseId = `resp_${uid()}`;
  const events = buildStreamingResponseEvents(responseId, model, previousResponseId, metadata);
  res.write(events.created());
  res.write(events.inProgress());

  let fullText = "";
  let inThink = false;
  let messageStarted = false;
  let completionSent = false;
  const toolCalls = new Map();
  let outputIndex = 0;
  let textOutputIdx = -1;
  let buffer = "";
  let streamOutput = null;
  const decoder = new TextDecoder();

  for await (const chunk of upstreamRes.body) {
    buffer += decoder.decode(chunk, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop();

    for (const line of lines) {
      if (!line.startsWith("data: ")) continue;
      const data = line.slice(6).trim();
      if (data === "[DONE]") {
        if (!completionSent) {
          completionSent = true;
          streamOutput = sendCompletion(res, events, responseId, model, fullText, toolCalls, outputIndex, textOutputIdx, null, null, previousResponseId, metadata);
        }
        continue;
      }

      let parsed;
      try {
        parsed = JSON.parse(data);
      } catch {
        continue;
      }

      const delta = parsed.choices?.[0]?.delta;
      const finishReason = parsed.choices?.[0]?.finish_reason;
      if (!delta && !finishReason) continue;

      if (delta?.tool_calls) {
        for (const tc of delta.tool_calls) {
          const idx = tc.index ?? 0;
          const tcOutIdx = (messageStarted && textOutputIdx === 0) ? outputIndex + idx + 1 : outputIndex + idx;
          if (!toolCalls.has(idx)) {
            const callId = tc.id || `call_${uid()}`;
            const fcId = `fc_${uid()}`;
            toolCalls.set(idx, { id: fcId, callId, name: tc.function?.name || "", arguments: "", outputIdx: tcOutIdx });
            res.write(events.outputItemAdded(tcOutIdx, {
              type: "function_call",
              id: fcId,
              call_id: callId,
              name: tc.function?.name || "",
              arguments: "",
              status: "in_progress",
            }));
          }
          if (tc.function?.arguments) {
            const tcData = toolCalls.get(idx);
            tcData.arguments += tc.function.arguments;
            res.write(events.fnCallArgsDelta(tcData.outputIdx, tcData.callId, tc.function.arguments));
          }
        }
        if (finishReason && !completionSent) {
          completionSent = true;
          streamOutput = sendCompletion(res, events, responseId, model, fullText, toolCalls, outputIndex, textOutputIdx, finishReason, parsed.usage, previousResponseId, metadata);
        }
        continue;
      }

      if (delta?.reasoning_content) continue;

      if (delta?.content) {
        let text = delta.content;
        if (text.includes("<think>")) { inThink = true; text = text.replace(/<think>/g, ""); }
        if (text.includes("</think>")) { inThink = false; text = text.replace(/<\/think>/g, ""); }
        if (inThink || !text) continue;

        if (!messageStarted) {
          messageStarted = true;
          textOutputIdx = outputIndex + toolCalls.size;
          res.write(events.outputItemAdded(textOutputIdx, {
            type: "message",
            id: `msg_${uid()}`,
            status: "in_progress",
            role: "assistant",
            content: [],
          }));
          res.write(events.contentPartAdded(textOutputIdx, 0, { type: "output_text", text: "", annotations: [] }));
        }

        fullText += text;
        res.write(events.textDelta(textOutputIdx, 0, text));
      }

      if (finishReason && !completionSent) {
        completionSent = true;
        streamOutput = sendCompletion(res, events, responseId, model, fullText, toolCalls, outputIndex, textOutputIdx, finishReason, parsed.usage, previousResponseId, metadata);
      }
    }
  }

  if (!completionSent) {
    completionSent = true;
    const wasGenerating = fullText.length > 0 || toolCalls.size > 0;
    const fallbackReason = wasGenerating ? "length" : "stop";
    console.warn(`[proxy] stream ended without finish_reason (wasGenerating=${wasGenerating}, reason=${fallbackReason})`);
    streamOutput = sendCompletion(res, events, responseId, model, fullText, toolCalls, outputIndex, textOutputIdx, fallbackReason, null, previousResponseId, metadata);
  }

  res.end();
  return { responseId, output: streamOutput || [] };
}

function sendCompletion(res, events, responseId, model, fullText, toolCalls, outputIndex, textOutputIdx, finishReason, usage, previousResponseId, metadata) {
  for (const [idx, tc] of toolCalls) {
    const tcIdx = tc.outputIdx != null ? tc.outputIdx : outputIndex + idx;
    res.write(events.fnCallArgsDone(tcIdx, tc.callId, tc.arguments));
    res.write(events.outputItemDone(tcIdx, {
      type: "function_call",
      id: tc.id,
      call_id: tc.callId,
      name: tc.name,
      arguments: tc.arguments,
      status: "completed",
    }));
  }

  const msgOutIdx = textOutputIdx >= 0 ? textOutputIdx : outputIndex + toolCalls.size;
  const trimmed = fullText.trim();
  if (trimmed) {
    const donePart = { type: "output_text", text: trimmed, annotations: [] };
    res.write(events.textDone(msgOutIdx, 0, trimmed));
    res.write(events.contentPartDone(msgOutIdx, 0, donePart));
    res.write(events.outputItemDone(msgOutIdx, {
      type: "message",
      id: `msg_${uid()}`,
      status: "completed",
      role: "assistant",
      content: [donePart],
    }));
  }

  const outputItems = [];
  for (const [idx, tc] of toolCalls) {
    const tcIdx = tc.outputIdx != null ? tc.outputIdx : outputIndex + idx;
    outputItems.push({
      sortIdx: tcIdx,
      item: {
        type: "function_call",
        id: tc.id,
        call_id: tc.callId,
        name: tc.name,
        arguments: tc.arguments,
        status: "completed",
      },
    });
  }
  if (trimmed) {
    outputItems.push({
      sortIdx: msgOutIdx,
      item: {
        type: "message",
        id: `msg_${uid()}`,
        status: "completed",
        role: "assistant",
        content: [{ type: "output_text", text: trimmed, annotations: [] }],
      },
    });
  }
  outputItems.sort((a, b) => a.sortIdx - b.sortIdx);
  const finalOutput = outputItems.map((o) => o.item);

  let status = "completed";
  let incompleteDetails = null;
  if (finishReason === "length") {
    status = "incomplete";
    incompleteDetails = { reason: "max_output_tokens" };
  }

  const finalResponse = {
    id: responseId,
    object: "response",
    created_at: Math.floor(Date.now() / 1000),
    status,
    model,
    output: finalOutput,
    previous_response_id: previousResponseId || null,
    metadata: metadata || {},
    usage: translateUsage(usage),
    incomplete_details: incompleteDetails,
  };

  res.write(events.completed(finalResponse));
  return finalOutput;
}

function sendResponseAsStream(res, response) {
  res.writeHead(200, {
    "Content-Type": "text/event-stream",
    "Cache-Control": "no-cache",
    Connection: "keep-alive",
  });

  const events = buildStreamingResponseEvents(response.id, response.model, response.previous_response_id, response.metadata);
  res.write(events.created());
  res.write(events.inProgress());

  for (let i = 0; i < response.output.length; i++) {
    const item = response.output[i];
    if (item.type === "function_call") {
      res.write(events.outputItemAdded(i, { ...item, status: "in_progress", arguments: "" }));
      res.write(events.fnCallArgsDelta(i, item.call_id, item.arguments));
      res.write(events.fnCallArgsDone(i, item.call_id, item.arguments));
      res.write(events.outputItemDone(i, item));
    } else if (item.type === "message") {
      res.write(events.outputItemAdded(i, { ...item, status: "in_progress", content: [] }));
      for (let ci = 0; ci < item.content.length; ci++) {
        const part = item.content[ci];
        if (part.type === "output_text") {
          res.write(events.contentPartAdded(i, ci, { type: "output_text", text: "", annotations: [] }));
          const text = part.text;
          for (let c = 0; c < text.length; c += 80) {
            res.write(events.textDelta(i, ci, text.slice(c, c + 80)));
          }
          res.write(events.textDone(i, ci, text));
          res.write(events.contentPartDone(i, ci, part));
        }
      }
      res.write(events.outputItemDone(i, item));
    }
  }

  res.write(events.completed(response));
  res.end();
}

// --- Generic upstream helpers ---

function sendJson(res, statusCode, payload) {
  res.writeHead(statusCode, { "Content-Type": "application/json" });
  res.end(JSON.stringify(payload));
}

async function readJsonBody(req, res) {
  let rawBody = "";
  for await (const chunk of req) rawBody += chunk;
  try {
    return JSON.parse(rawBody);
  } catch {
    sendJson(res, 400, { error: "Invalid JSON" });
    return null;
  }
}

async function sendUpstreamError(upstreamRes, res) {
  const errText = await upstreamRes.text();
  console.error(`[proxy] upstream error: ${upstreamRes.status} ${errText}`);
  if (!res.headersSent) {
    res.writeHead(upstreamRes.status, { "Content-Type": upstreamRes.headers.get("content-type") || "application/json" });
    res.end(errText);
  }
}

async function pipeResponsesStreamAndCapture(upstreamRes, res, onCompleted) {
  res.writeHead(upstreamRes.status, {
    "Content-Type": "text/event-stream",
    "Cache-Control": "no-cache",
    Connection: "keep-alive",
  });

  let buffer = "";
  const decoder = new TextDecoder();

  const handleBlock = (block) => {
    const lines = block.split("\n");
    let eventType = "";
    const dataLines = [];

    for (const line of lines) {
      if (line.startsWith("event:")) eventType = line.slice(6).trim();
      else if (line.startsWith("data:")) dataLines.push(line.slice(5).trimStart());
    }

    const data = dataLines.join("\n");
    if (!data || data === "[DONE]") return;

    try {
      const parsed = JSON.parse(data);
      if (eventType === "response.completed" || parsed.type === "response.completed") {
        onCompleted(parsed.response || parsed);
      }
    } catch {
      // Ignore parse failures in streamed event capture; stream still passes through.
    }
  };

  for await (const chunk of upstreamRes.body) {
    res.write(chunk);
    buffer += decoder.decode(chunk, { stream: true }).replace(/\r\n/g, "\n");

    let splitIdx;
    while ((splitIdx = buffer.indexOf("\n\n")) !== -1) {
      const block = buffer.slice(0, splitIdx);
      buffer = buffer.slice(splitIdx + 2);
      handleBlock(block);
    }
  }

  if (buffer.trim()) handleBlock(buffer);
  res.end();
}

async function forwardOpenAIResponses(body, res, originalInput, originalPreviousResponseId) {
  const upstreamRes = await fetch(`${OPENAI_BASE}/responses`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${OPENAI_KEY}`,
    },
    body: JSON.stringify(body),
  });

  if (!upstreamRes.ok) {
    await sendUpstreamError(upstreamRes, res);
    return;
  }

  if (body.stream) {
    await pipeResponsesStreamAndCapture(upstreamRes, res, (completedResponse) => {
      if (completedResponse?.id && Array.isArray(completedResponse.output)) {
        storeResponse(completedResponse.id, {
          provider: "openai",
          input: originalInput,
          output: completedResponse.output,
          previousResponseId: originalPreviousResponseId || null,
        });
      }
    });
    return;
  }

  const response = await upstreamRes.json();
  if (response?.id && Array.isArray(response.output)) {
    storeResponse(response.id, {
      provider: "openai",
      input: originalInput,
      output: response.output,
      previousResponseId: originalPreviousResponseId || null,
    });
  }
  sendJson(res, upstreamRes.status, response);
}

async function forwardOpenAIChatCompletions(body, res) {
  const upstreamRes = await fetch(`${OPENAI_BASE}/chat/completions`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${OPENAI_KEY}`,
    },
    body: JSON.stringify(body),
  });

  if (!upstreamRes.ok) {
    await sendUpstreamError(upstreamRes, res);
    return;
  }

  if (body.stream) {
    res.writeHead(upstreamRes.status, {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache",
      Connection: "keep-alive",
    });
    for await (const chunk of upstreamRes.body) res.write(chunk);
    res.end();
    return;
  }

  const response = await upstreamRes.json();
  sendJson(res, upstreamRes.status, response);
}

// --- MiniMax handlers ---

async function handleMinimaxResponses(body, res, originalInput) {
  if (!MINIMAX_KEY) {
    sendJson(res, 400, { error: { message: "MINIMAX_API_KEY is not configured" } });
    return;
  }

  const originalPreviousResponseId = body.previous_response_id || null;
  maybeResolvePreviousResponseChain(body, "minimax");

  if (originalPreviousResponseId) {
    const prevStored = responseStore.get(originalPreviousResponseId);
    const consecutiveTc = prevStored?.consecutiveToolCalls || 0;
    if (consecutiveTc >= MAX_CONSECUTIVE_TOOL_CALLS) {
      console.warn(`[proxy] CIRCUIT BREAKER: ${consecutiveTc} consecutive tool-call-only responses detected — injecting stop-loop nudge`);
      const nudge = {
        type: "message",
        role: "user",
        content: [{
          type: "input_text",
          text: `[SYSTEM: You have made ${consecutiveTc} consecutive tool calls without responding to the user. You MUST now stop making tool calls and provide a text response summarizing your progress, findings, and any remaining work. Do NOT make any more tool calls in this response.]`,
        }],
      };
      const currentInput = normalizeInputToArray(body.input);
      body.input = [...currentInput, nudge];
    } else if (consecutiveTc >= Math.floor(MAX_CONSECUTIVE_TOOL_CALLS * 0.75)) {
      console.warn(`[proxy] tool-call loop warning: ${consecutiveTc}/${MAX_CONSECUTIVE_TOOL_CALLS} consecutive tool-call responses`);
    }
  }

  const hasWebSearch = body.tools?.some((t) => t.type === "web_search" || t.type === "web_search_preview");
  const hasFunctionTools = body.tools?.some((t) => t.type === "function");
  const useOpenRouter = hasWebSearch && OPENROUTER_KEY;

  if (hasWebSearch && !OPENROUTER_KEY) {
    console.warn("[proxy] web_search requested but OPENROUTER_API_KEY not set — web search will be dropped");
  }

  const chatReq = responsesRequestToChatCompletions(body);
  chatReq.model = MINIMAX_MODELS[0] || "MiniMax-M2.7";
  const isStream = chatReq.stream;

  let upstreamUrl;
  let upstreamKey;
  let routeLabel;

  if (useOpenRouter) {
    chatReq.model = OPENROUTER_SEARCH_MODEL;
    chatReq.plugins = [{ id: "web", max_results: 5 }];
    delete chatReq.reasoning_split;
    if (hasFunctionTools && (!chatReq.tools || chatReq.tools.length === 0)) {
      const supported = body.tools.filter((t) => t.type === "function");
      if (supported.length > 0) {
        chatReq.tools = supported.map((t) => {
          if (!t.function) {
            return {
              type: "function",
              function: { name: t.name, description: t.description, parameters: t.parameters },
            };
          }
          return t;
        });
      }
    }
    upstreamUrl = `${OPENROUTER_BASE}/chat/completions`;
    upstreamKey = OPENROUTER_KEY;
    routeLabel = `openrouter(${OPENROUTER_SEARCH_MODEL})`;
  } else {
    upstreamUrl = `${MINIMAX_BASE}/chat/completions`;
    upstreamKey = MINIMAX_KEY;
    routeLabel = `minimax(${chatReq.model})`;
    chatReq.reasoning_split = true;
  }

  if (originalPreviousResponseId) {
    const prevStored = responseStore.get(originalPreviousResponseId);
    const consecutiveTc = prevStored?.consecutiveToolCalls || 0;
    if (consecutiveTc >= MAX_CONSECUTIVE_TOOL_CALLS + 3) {
      console.warn("[proxy] HARD CIRCUIT BREAKER: stripping all tools to force text response");
      delete chatReq.tools;
      delete chatReq.tool_choice;
    }
  }

  const hasConversationUrls = conversationHasUrls(chatReq.messages);
  if (hasConversationUrls) {
    chatReq.tools = ensureWebFetchTool(chatReq.tools);
    chatReq.messages = ensureWebFetchHint(chatReq.messages);
  }

  console.log(
    `[proxy] ${routeLabel} | stream=${isStream} | messages=${chatReq.messages.length}${hasWebSearch ? " | web_search" : ""}${hasConversationUrls ? " | web_fetch_injected" : ""} | roles=[${chatReq.messages.map((m) => m.role + (m.tool_calls ? "(tc)" : "")).join(",")}]`
  );

  if (hasConversationUrls) {
    let loopMessages = [...chatReq.messages];
    let finalCcResponse = null;
    let fetchLoopCount = 0;
    const fetchCache = new Map();
    let prevFetchUrls = "";

    for (let loop = 0; loop <= MAX_FETCH_LOOPS; loop++) {
      const loopReq = { ...chatReq, messages: loopMessages, stream: false };
      const upstreamRes = await fetch(upstreamUrl, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${upstreamKey}`,
        },
        body: JSON.stringify(loopReq),
      });

      if (!upstreamRes.ok) {
        await sendUpstreamError(upstreamRes, res);
        return;
      }

      const ccResponse = await upstreamRes.json();
      const msg = ccResponse.choices?.[0]?.message;
      const webFetchCalls = (msg?.tool_calls || []).filter((tc) => tc.function?.name === "web_fetch");
      const currentFetchUrls = webFetchCalls.map((tc) => {
        try { return JSON.parse(tc.function.arguments).url; }
        catch { return ""; }
      }).sort().join("|");
      const isStuckLoop = webFetchCalls.length > 0 && currentFetchUrls === prevFetchUrls;

      if (webFetchCalls.length === 0 || loop === MAX_FETCH_LOOPS || isStuckLoop) {
        if (isStuckLoop) {
          console.warn(`[proxy] web_fetch loop stuck — model re-requested same URL(s), breaking early at loop ${loop + 1}`);
        }
        if (loop === MAX_FETCH_LOOPS && webFetchCalls.length > 0) {
          console.warn(`[proxy] web_fetch MAX_FETCH_LOOPS (${MAX_FETCH_LOOPS}) exhausted — model still requesting fetches, stripping them`);
        }
        if (msg?.tool_calls) {
          msg.tool_calls = msg.tool_calls.filter((tc) => tc.function?.name !== "web_fetch");
          if (msg.tool_calls.length === 0) {
            delete msg.tool_calls;
            if (ccResponse.choices[0].finish_reason === "tool_calls") {
              ccResponse.choices[0].finish_reason = "stop";
            }
          }
        }
        finalCcResponse = ccResponse;
        fetchLoopCount = loop;
        break;
      }

      prevFetchUrls = currentFetchUrls;
      console.log(`[proxy] executing ${webFetchCalls.length} web_fetch call(s) (loop ${loop + 1}/${MAX_FETCH_LOOPS})`);
      const results = await Promise.all(webFetchCalls.map(async (tc) => {
        const fetchUrl = (() => {
          try { return JSON.parse(tc.function.arguments).url; }
          catch { return "unknown"; }
        })();
        if (fetchCache.has(fetchUrl)) {
          console.log(`[proxy] web_fetch ${fetchUrl} -> ${fetchCache.get(fetchUrl).length} chars (cached)`);
          return { role: "tool", tool_call_id: tc.id, content: fetchCache.get(fetchUrl) };
        }
        const content = await executeWebFetch(tc.function.arguments);
        fetchCache.set(fetchUrl, content);
        console.log(`[proxy] web_fetch ${fetchUrl} -> ${content.length} chars`);
        return { role: "tool", tool_call_id: tc.id, content };
      }));

      loopMessages = [
        ...loopMessages,
        { role: "assistant", content: null, tool_calls: webFetchCalls },
        ...results,
      ];
    }

    if (fetchLoopCount > 0) {
      console.log(`[proxy] web_fetch resolved after ${fetchLoopCount} loop(s)`);
    }

    const responsesResponse = chatCompletionToResponse(finalCcResponse, body.model, originalPreviousResponseId, body.metadata);
    storeResponse(responsesResponse.id, {
      provider: "minimax",
      input: originalInput,
      output: responsesResponse.output,
      previousResponseId: originalPreviousResponseId,
    });

    if (isStream) sendResponseAsStream(res, responsesResponse);
    else sendJson(res, 200, responsesResponse);
    return;
  }

  const upstreamRes = await fetch(upstreamUrl, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${upstreamKey}`,
    },
    body: JSON.stringify(chatReq),
  });

  if (!upstreamRes.ok) {
    await sendUpstreamError(upstreamRes, res);
    return;
  }

  if (isStream) {
    const { responseId: streamRespId, output: streamOutput } = await handleStreamingResponse(
      upstreamRes,
      res,
      body.model,
      originalPreviousResponseId,
      body.metadata
    );
    storeResponse(streamRespId, {
      provider: "minimax",
      input: originalInput,
      output: streamOutput,
      previousResponseId: originalPreviousResponseId,
    });
    return;
  }

  const ccResponse = await upstreamRes.json();
  const responsesResponse = chatCompletionToResponse(ccResponse, body.model, originalPreviousResponseId, body.metadata);
  storeResponse(responsesResponse.id, {
    provider: "minimax",
    input: originalInput,
    output: responsesResponse.output,
    previousResponseId: originalPreviousResponseId,
  });
  sendJson(res, 200, responsesResponse);
}

async function handleMinimaxChatCompletions(body, res) {
  if (!MINIMAX_KEY) {
    sendJson(res, 400, { error: { message: "MINIMAX_API_KEY is not configured" } });
    return;
  }

  body.model = body.model || MINIMAX_MODELS[0] || "MiniMax-M2.7";
  const isStream = body.stream || false;

  let messages = body.messages || [];
  const fixed = [];
  for (let i = 0; i < messages.length; i++) {
    const msg = messages[i];
    if (msg === null) {
      continue;
    } else if (msg.role === "assistant" && msg.tool_calls) {
      fixed.push(msg);
      const callIds = new Set(msg.tool_calls.map((tc) => tc.id));
      for (let j = i + 1; j < messages.length; j++) {
        if (messages[j] && messages[j].role === "tool" && callIds.has(messages[j].tool_call_id)) {
          fixed.push(messages[j]);
          messages[j] = null;
        }
      }
    } else if (msg.role === "tool") {
      const lastTc = [...fixed].reverse().find((m) => m.role === "assistant" && m.tool_calls);
      if (lastTc) {
        let insertIdx = fixed.indexOf(lastTc) + 1;
        while (insertIdx < fixed.length && fixed[insertIdx].role === "tool") insertIdx++;
        fixed.splice(insertIdx, 0, msg);
        messages[i] = null;
      }
    } else {
      fixed.push(msg);
    }
  }

  const merged = [];
  for (const msg of fixed) {
    const prev = merged[merged.length - 1];
    if (prev && prev.role === msg.role && msg.role === "user" && typeof prev.content === "string" && typeof msg.content === "string") {
      prev.content += "\n\n" + msg.content;
    } else if (prev && prev.role === msg.role && msg.role === "assistant" && !prev.tool_calls && !msg.tool_calls && typeof prev.content === "string" && typeof msg.content === "string") {
      prev.content += "\n\n" + msg.content;
    } else if (prev && prev.role === "assistant" && msg.role === "assistant" && !prev.tool_calls && msg.tool_calls) {
      merged[merged.length - 1] = msg;
    } else if (prev && prev.role === "assistant" && msg.role === "assistant" && prev.tool_calls && !msg.tool_calls) {
      // Drop text-only assistant after tool calls.
    } else {
      merged.push(msg);
    }
  }

  const validated = [];
  for (const msg of merged) {
    if (msg.role === "tool") {
      const prev = validated[validated.length - 1];
      if (prev && (prev.role === "tool" || (prev.role === "assistant" && prev.tool_calls))) {
        validated.push(msg);
      }
    } else {
      validated.push(msg);
    }
  }

  for (const msg of validated) {
    if (msg.role === "assistant" && msg.tool_calls) {
      for (const tc of msg.tool_calls) {
        if (!tc.function) continue;
        const args = tc.function.arguments;
        if (args === undefined || args === null || args === "") {
          tc.function.arguments = "{}";
        } else if (typeof args !== "string") {
          tc.function.arguments = JSON.stringify(args);
        } else {
          try {
            JSON.parse(args);
          } catch {
            console.warn(`[proxy] invalid tool_call arguments for ${tc.function.name} (id: ${tc.id}), wrapping as JSON`);
            tc.function.arguments = JSON.stringify({ input: args });
          }
        }
      }
    }
    if (msg.role === "tool" && typeof msg.content !== "string") {
      msg.content = JSON.stringify(msg.content);
    }
  }

  body.messages = validated;
  body.reasoning_split = true;
  if (!body.max_tokens) body.max_tokens = 16384;

  const ccHasUrls = conversationHasUrls(validated);

  if (ccHasUrls) {
    body.tools = ensureWebFetchTool(body.tools);
    body.messages = ensureWebFetchHint(body.messages);
  }

  console.log(`[proxy] chat/completions minimax(${body.model}) | stream=${isStream} | messages=${body.messages.length}${ccHasUrls ? " | web_fetch_injected" : ""} | roles=[${body.messages.map((m) => m.role + (m.tool_calls ? "(tc)" : "")).join(",")}]`);

  if (ccHasUrls) {
    let loopMessages = [...body.messages];
    let finalCcResponse = null;
    let fetchLoopCount = 0;
    const fetchCache = new Map();
    let prevFetchUrls = "";

    for (let loop = 0; loop <= MAX_FETCH_LOOPS; loop++) {
      const loopBody = { ...body, messages: loopMessages, stream: false };
      const upstreamRes = await fetch(`${MINIMAX_BASE}/chat/completions`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${MINIMAX_KEY}`,
        },
        body: JSON.stringify(loopBody),
      });

      if (!upstreamRes.ok) {
        await sendUpstreamError(upstreamRes, res);
        return;
      }

      const ccResponse = await upstreamRes.json();
      const msg = ccResponse.choices?.[0]?.message;
      const webFetchCalls = (msg?.tool_calls || []).filter((tc) => tc.function?.name === "web_fetch");
      const currentFetchUrls = webFetchCalls.map((tc) => {
        try { return JSON.parse(tc.function.arguments).url; }
        catch { return ""; }
      }).sort().join("|");
      const isStuckLoop = webFetchCalls.length > 0 && currentFetchUrls === prevFetchUrls;

      if (webFetchCalls.length === 0 || loop === MAX_FETCH_LOOPS || isStuckLoop) {
        if (isStuckLoop) {
          console.warn(`[proxy] cc: web_fetch loop stuck — model re-requested same URL(s), breaking early at loop ${loop + 1}`);
        }
        if (loop === MAX_FETCH_LOOPS && webFetchCalls.length > 0) {
          console.warn(`[proxy] cc: web_fetch MAX_FETCH_LOOPS (${MAX_FETCH_LOOPS}) exhausted — stripping remaining fetches`);
        }
        if (msg?.tool_calls) {
          msg.tool_calls = msg.tool_calls.filter((tc) => tc.function?.name !== "web_fetch");
          if (msg.tool_calls.length === 0) {
            delete msg.tool_calls;
            if (ccResponse.choices[0].finish_reason === "tool_calls") {
              ccResponse.choices[0].finish_reason = "stop";
            }
          }
        }
        finalCcResponse = ccResponse;
        fetchLoopCount = loop;
        break;
      }

      prevFetchUrls = currentFetchUrls;
      console.log(`[proxy] cc: executing ${webFetchCalls.length} web_fetch call(s) (loop ${loop + 1}/${MAX_FETCH_LOOPS})`);
      const results = await Promise.all(webFetchCalls.map(async (tc) => {
        const fetchUrl = (() => {
          try { return JSON.parse(tc.function.arguments).url; }
          catch { return "unknown"; }
        })();
        if (fetchCache.has(fetchUrl)) {
          console.log(`[proxy] cc: web_fetch ${fetchUrl} -> ${fetchCache.get(fetchUrl).length} chars (cached)`);
          return { role: "tool", tool_call_id: tc.id, content: fetchCache.get(fetchUrl) };
        }
        const content = await executeWebFetch(tc.function.arguments);
        fetchCache.set(fetchUrl, content);
        console.log(`[proxy] cc: web_fetch ${fetchUrl} -> ${content.length} chars`);
        return { role: "tool", tool_call_id: tc.id, content };
      }));

      loopMessages = [
        ...loopMessages,
        { role: "assistant", content: null, tool_calls: webFetchCalls },
        ...results,
      ];
    }

    if (fetchLoopCount > 0) console.log(`[proxy] cc: web_fetch resolved after ${fetchLoopCount} loop(s)`);

    if (isStream) {
      res.writeHead(200, {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        Connection: "keep-alive",
      });
      const msg = finalCcResponse.choices?.[0]?.message;
      if (msg?.tool_calls) {
        for (let i = 0; i < msg.tool_calls.length; i++) {
          const tc = msg.tool_calls[i];
          res.write(`data: ${JSON.stringify({ choices: [{ index: 0, delta: { tool_calls: [{ index: i, id: tc.id, type: "function", function: { name: tc.function.name, arguments: "" } }] } }] })}\n\n`);
          res.write(`data: ${JSON.stringify({ choices: [{ index: 0, delta: { tool_calls: [{ index: i, function: { arguments: tc.function.arguments } }] } }] })}\n\n`);
        }
      }
      if (msg?.content) {
        res.write(`data: ${JSON.stringify({ choices: [{ index: 0, delta: { content: msg.content } }] })}\n\n`);
      }
      res.write(`data: ${JSON.stringify({ choices: [{ index: 0, delta: {}, finish_reason: finalCcResponse.choices[0].finish_reason }], usage: finalCcResponse.usage })}\n\n`);
      res.write("data: [DONE]\n\n");
      res.end();
      return;
    }

    sendJson(res, 200, finalCcResponse);
    return;
  }

  const upstreamRes = await fetch(`${MINIMAX_BASE}/chat/completions`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${MINIMAX_KEY}`,
    },
    body: JSON.stringify(body),
  });

  if (!upstreamRes.ok) {
    await sendUpstreamError(upstreamRes, res);
    return;
  }

  if (isStream) {
    res.writeHead(200, {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache",
      Connection: "keep-alive",
    });
    for await (const chunk of upstreamRes.body) res.write(chunk);
    res.end();
    return;
  }

  const data = await upstreamRes.json();
  sendJson(res, 200, data);
}

// --- HTTP server ---

const server = http.createServer(async (req, res) => {
  if (req.method === "GET" && (req.url === "/health" || req.url === "/")) {
    sendJson(res, 200, {
      status: "ok",
      proxy: "codex-minimax-proxy",
      providers: [...enabledProviders],
      default_provider: getFallbackProvider(),
    });
    return;
  }

  if ((req.method === "GET" || req.method === "POST") && req.url.startsWith("/cop")) {
    let url = "";
    let method = "GET";
    let body2 = null;
    let headers2 = {};

    if (req.method === "GET") {
      const parsed = new URL(req.url, "http://localhost");
      url = parsed.searchParams.get("url") || "";
    } else {
      const parsedBody = await readJsonBody(req, res);
      if (!parsedBody) return;
      url = parsedBody.url || "";
      method = parsedBody.method || "GET";
      body2 = parsedBody.body || null;
      headers2 = parsedBody.headers || {};
    }

    if (!url) {
      sendJson(res, 400, { error: "url parameter required" });
      return;
    }

    console.log(`[proxy] /cop ${method} ${url}`);
    const content = await executeWebFetch({ url, method, headers: headers2, body: body2 });
    res.writeHead(200, { "Content-Type": "text/plain; charset=utf-8" });
    res.end(content);
    return;
  }

  if (req.method === "GET" && (req.url === "/v1/models" || req.url === "/models")) {
    sendJson(res, 200, {
      object: "list",
      data: modelCatalog,
      default_provider: getFallbackProvider(),
    });
    return;
  }

  if (req.method === "POST" && (req.url === "/v1/responses" || req.url === "/responses")) {
    const body = await readJsonBody(req, res);
    if (!body) return;

    try {
      const provider = resolveProviderForModel(body.model);
      const originalInput = normalizeInputToArray(body.input);

      if (provider === "openai") {
        if (!OPENAI_KEY) {
          sendJson(res, 400, { error: { message: "OPENAI_API_KEY is not configured" } });
          return;
        }
        const originalPreviousResponseId = body.previous_response_id || null;
        maybeResolvePreviousResponseChain(body, "openai");
        console.log(`[proxy] responses openai(${body.model || OPENAI_MODELS[0] || "default"}) | stream=${!!body.stream}`);
        await forwardOpenAIResponses(body, res, originalInput, originalPreviousResponseId);
        return;
      }

      await handleMinimaxResponses(body, res, originalInput);
    } catch (err) {
      console.error("[proxy] responses route error:", err.message);
      if (!res.headersSent) sendJson(res, 500, { error: { message: err.message } });
    }
    return;
  }

  if (req.method === "POST" && (req.url === "/v1/chat/completions" || req.url === "/chat/completions")) {
    const body = await readJsonBody(req, res);
    if (!body) return;

    try {
      const provider = resolveProviderForModel(body.model);
      if (provider === "openai") {
        if (!OPENAI_KEY) {
          sendJson(res, 400, { error: { message: "OPENAI_API_KEY is not configured" } });
          return;
        }
        console.log(`[proxy] chat/completions openai(${body.model || OPENAI_MODELS[0] || "default"}) | stream=${!!body.stream}`);
        await forwardOpenAIChatCompletions(body, res);
        return;
      }

      await handleMinimaxChatCompletions(body, res);
    } catch (err) {
      console.error("[proxy] chat/completions route error:", err.message);
      if (!res.headersSent) sendJson(res, 500, { error: { message: err.message } });
    }
    return;
  }

  sendJson(res, 404, { error: "Not found. Use POST /v1/responses" });
});

server.timeout = 0;
server.keepAliveTimeout = 300000;
server.headersTimeout = 300000;
server.requestTimeout = 0;

server.listen(PORT, () => {
  console.log(`[codex-minimax-proxy] Listening on http://localhost:${PORT}`);
  console.log(`[codex-minimax-proxy] Default provider: ${getFallbackProvider()}`);
  console.log(`[codex-minimax-proxy] MiniMax: ${MINIMAX_KEY ? `${MINIMAX_BASE} | models=${providerModels.minimax.join(", ")}` : "DISABLED"}`);
  console.log(`[codex-minimax-proxy] OpenAI:  ${OPENAI_KEY ? `${OPENAI_BASE} | models=${providerModels.openai.join(", ")}` : "DISABLED"}`);
  console.log(`[codex-minimax-proxy] Search:  ${OPENROUTER_KEY ? `OpenRouter (${OPENROUTER_SEARCH_MODEL})` : "DISABLED (no OPENROUTER_API_KEY)"}`);
  console.log(`[codex-minimax-proxy] GitHub:  ${GITHUB_TOKEN ? "authenticated" : "anonymous (set GITHUB_TOKEN or install gh CLI)"}`);
});
