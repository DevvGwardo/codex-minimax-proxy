import http from "node:http";
import crypto from "node:crypto";

process.on("uncaughtException", (err) => {
  console.error("[proxy] uncaught exception:", err.message);
});
process.on("unhandledRejection", (err) => {
  console.error("[proxy] unhandled rejection:", err.message || err);
});

const PORT = process.env.PROXY_PORT || 4000;
const MINIMAX_BASE = process.env.MINIMAX_BASE_URL || "https://api.minimax.io/v1";
const MINIMAX_KEY = process.env.MINIMAX_API_KEY;
const OPENROUTER_KEY = process.env.OPENROUTER_API_KEY;
const OPENROUTER_BASE = "https://openrouter.ai/api/v1";
const OPENROUTER_SEARCH_MODEL = process.env.OPENROUTER_SEARCH_MODEL || "nvidia/nemotron-3-super-120b-a12b:free";

if (!MINIMAX_KEY) {
  console.error("MINIMAX_API_KEY env var is required");
  process.exit(1);
}
if (!OPENROUTER_KEY) {
  console.warn("[proxy] OPENROUTER_API_KEY not set — web search requests will be skipped");
}

// --- Response store for previous_response_id support ---

const responseStore = new Map();
const STORE_TTL = 60 * 60 * 1000; // 1 hour
const STORE_MAX = 500;
const MAX_CONSECUTIVE_TOOL_CALLS = 20; // circuit breaker threshold

function storeResponse(id, data) {
  // Evict expired entries periodically
  if (responseStore.size >= STORE_MAX) {
    const now = Date.now();
    for (const [key, val] of responseStore) {
      if (now - val.storedAt > STORE_TTL) responseStore.delete(key);
    }
    // If still over limit, remove oldest
    if (responseStore.size >= STORE_MAX) {
      const oldest = responseStore.keys().next().value;
      responseStore.delete(oldest);
    }
  }

  // Track consecutive tool-call-only responses for circuit breaker
  const isToolCallOnly = Array.isArray(data.output) &&
    data.output.length > 0 &&
    data.output.every(o => o.type === "function_call");

  let consecutiveToolCalls = 0;
  if (isToolCallOnly && data.previousResponseId) {
    const prev = responseStore.get(data.previousResponseId);
    consecutiveToolCalls = (prev?.consecutiveToolCalls || 0) + 1;
  }

  responseStore.set(id, { ...data, storedAt: Date.now(), consecutiveToolCalls });
  console.log(`[proxy] stored response ${id} (store size: ${responseStore.size}${consecutiveToolCalls > 0 ? `, consecutive_tc: ${consecutiveToolCalls}` : ""})`);
}

function resolveResponseChain(previousResponseId) {
  // Walk the chain backwards, collecting input+output from each stored response
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
    chain.unshift(stored); // prepend so oldest is first
    currentId = stored.previousResponseId;
  }

  // Flatten: for each stored response, emit its input then its output
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

// --- Request translation: Responses API -> Chat Completions ---

function responsesRequestToChatCompletions(body) {
  const messages = [];

  // instructions -> user message (MiniMax doesn't support system role)
  if (body.instructions) {
    messages.push({ role: "user", content: "[System Instructions] " + body.instructions + "\n\nNote: Be efficient with tool calls. Avoid repeating the same tool call unnecessarily." });
  }

  // input -> messages
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
            if (block.type === "input_image")
              return { type: "image_url", image_url: { url: block.image_url || block.url } };
            return block;
          });
          // If all blocks are text, simplify to string
          if (content.length === 1 && content[0].type === "text") {
            content = content[0].text;
          }
        }

        if (pendingToolCalls.length > 0 && role === "assistant") {
          // Assistant text + pending tool calls from same turn — combine into one message.
          // Drop the text content (set null) to prevent MiniMax from echoing it.
          // The text is typically transitional ("let me check...") and was already shown to the user.
          messages.push({ role: "assistant", content: null, tool_calls: pendingToolCalls });
          pendingToolCalls = [];
        } else {
          // Flush any pending tool calls into a separate assistant message
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
        // Flush pending tool calls first
        if (pendingToolCalls.length > 0) {
          messages.push({ role: "assistant", content: null, tool_calls: pendingToolCalls });
          pendingToolCalls = [];
        }
        messages.push({ role: "tool", tool_call_id: item.call_id, content: item.output });
      } else if (item.type === "reasoning") {
        // Skip reasoning items
      }
    }

    // Flush remaining tool calls
    if (pendingToolCalls.length > 0) {
      messages.push({ role: "assistant", content: null, tool_calls: pendingToolCalls });
    }
  }

  // Fix message ordering for MiniMax:
  // 1. Tool results must directly follow their assistant(tc) message
  // 2. No consecutive same-role messages (except tool after tool)
  // Strategy: collect tool_call_ids from each assistant(tc), then group tool results after it
  const fixed = [];
  for (let i = 0; i < messages.length; i++) {
    const msg = messages[i];

    if (msg === null) {
      continue;
    } else if (msg.role === "assistant" && msg.tool_calls) {
      fixed.push(msg);
      // Collect all tool_call_ids from this assistant message
      const callIds = new Set(msg.tool_calls.map((tc) => tc.id));
      // Gather matching tool results from anywhere ahead in the array
      for (let j = i + 1; j < messages.length; j++) {
        if (messages[j].role === "tool" && callIds.has(messages[j].tool_call_id)) {
          fixed.push(messages[j]);
          messages[j] = null; // mark as consumed
        }
      }
    } else if (msg.role === "tool") {
      // Orphan tool message — skip or attach to last assistant(tc)
      // Find last assistant with tool_calls and append
      const lastTc = [...fixed].reverse().find((m) => m.role === "assistant" && m.tool_calls);
      if (lastTc) {
        // Insert after the last tool message following that assistant
        let insertIdx = fixed.indexOf(lastTc) + 1;
        while (insertIdx < fixed.length && fixed[insertIdx].role === "tool") insertIdx++;
        fixed.splice(insertIdx, 0, msg);
        messages[i] = null; // mark as consumed
      }
      // Otherwise drop it — can't send orphan tool results
    } else {
      fixed.push(msg);
    }
  }

  // Merge consecutive same-role messages (user+user, assistant+assistant, and assistant+assistant(tc))
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
      // Merge consecutive plain-text assistant messages (no tool_calls)
      prev.content += "\n\n" + msg.content;
    } else if (
      prev &&
      prev.role === "assistant" &&
      msg.role === "assistant" &&
      !prev.tool_calls &&
      msg.tool_calls
    ) {
      // Drop text-only assistant, keep assistant(tc) — prevents consecutive assistant messages.
      // Text content is not copied to avoid MiniMax echoing it (causes duplicate messages).
      merged[merged.length - 1] = msg;
    } else if (
      prev &&
      prev.role === "assistant" &&
      msg.role === "assistant" &&
      prev.tool_calls &&
      !msg.tool_calls
    ) {
      // Drop text-only assistant that follows tool calls — already displayed to user.
      // Keep the assistant(tc) message as-is with content: null.
    } else {
      merged.push(msg);
    }
  }

  // Truncate large tool outputs in older messages to save context space.
  // Terminal commands (cat heredocs, compile output, grep) can produce huge content
  // that eats through the context window fast. Keep recent ones full.
  const TOOL_OUTPUT_MAX = 2000; // chars
  const KEEP_RECENT_FULL = 10; // keep last N messages at full size
  for (let i = 0; i < Math.max(0, merged.length - KEEP_RECENT_FULL); i++) {
    const msg = merged[i];
    if (msg.role === "tool" && typeof msg.content === "string" && msg.content.length > TOOL_OUTPUT_MAX) {
      msg.content = msg.content.slice(0, TOOL_OUTPUT_MAX) + "\n...[output truncated, " + (msg.content.length - TOOL_OUTPUT_MAX) + " chars removed]";
    }
  }

  // Context trimming: keep first 2 + last 48 messages, drop middle
  const MAX_MESSAGES = 55;
  let finalMessages = merged;

  if (merged.length > MAX_MESSAGES) {
    const head = merged.slice(0, 2);
    let tail = merged.slice(-(MAX_MESSAGES - 3));
    // Ensure tail starts with user or assistant (not tool)
    while (tail.length > 0 && tail[0].role === "tool") tail.shift();
    finalMessages = [
      ...head,
      { role: "user", content: "[Earlier conversation trimmed. Do not repeat previous statements or tool calls you already made. Continue with the current task. If you have enough information, respond to the user instead of making more tool calls.]" },
      ...tail,
    ];
    console.log(`[proxy] trimmed ${merged.length} -> ${finalMessages.length} messages`);
  }

  // FINAL VALIDATION: ensure tool messages always follow assistant(tc)
  // This is critical for MiniMax — any violation causes a 400 error
  const validated = [];
  for (let i = 0; i < finalMessages.length; i++) {
    const msg = finalMessages[i];
    if (msg.role === "tool") {
      // Only include if previous message is assistant(tc) or another tool
      const prev = validated[validated.length - 1];
      if (prev && (prev.role === "tool" || (prev.role === "assistant" && prev.tool_calls))) {
        validated.push(msg);
      }
      // Otherwise drop this orphan tool message
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
  // Default to 16384 max tokens to prevent premature truncation
  req.max_tokens = body.max_output_tokens || 16384;

  // Tools translation: flat -> nested, filter out unsupported types
  if (body.tools && body.tools.length > 0) {
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

  if (body.reasoning?.effort) {
    req.reasoning_effort = body.reasoning.effort;
  }

  if (body.parallel_tool_calls != null) {
    req.parallel_tool_calls = body.parallel_tool_calls;
  }

  return req;
}

// --- Response translation: Chat Completions -> Responses API ---

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

  // Handle tool calls
  if (msg.tool_calls && msg.tool_calls.length > 0) {
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

  // Handle text content (strip <think> tags)
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

  // Handle refusal
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

// --- Streaming translation ---

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
    created: () =>
      `event: response.created\ndata: ${JSON.stringify({ type: "response.created", response: baseResponse })}\n\n`,
    inProgress: () =>
      `event: response.in_progress\ndata: ${JSON.stringify({ type: "response.in_progress", response: baseResponse })}\n\n`,
    outputItemAdded: (index, item) =>
      `event: response.output_item.added\ndata: ${JSON.stringify({ type: "response.output_item.added", output_index: index, item })}\n\n`,
    contentPartAdded: (outIdx, contentIdx, part) =>
      `event: response.content_part.added\ndata: ${JSON.stringify({ type: "response.content_part.added", output_index: outIdx, content_index: contentIdx, part })}\n\n`,
    textDelta: (outIdx, contentIdx, delta) =>
      `event: response.output_text.delta\ndata: ${JSON.stringify({ type: "response.output_text.delta", output_index: outIdx, content_index: contentIdx, delta })}\n\n`,
    textDone: (outIdx, contentIdx, text) =>
      `event: response.output_text.done\ndata: ${JSON.stringify({ type: "response.output_text.done", output_index: outIdx, content_index: contentIdx, text })}\n\n`,
    contentPartDone: (outIdx, contentIdx, part) =>
      `event: response.content_part.done\ndata: ${JSON.stringify({ type: "response.content_part.done", output_index: outIdx, content_index: contentIdx, part })}\n\n`,
    outputItemDone: (outIdx, item) =>
      `event: response.output_item.done\ndata: ${JSON.stringify({ type: "response.output_item.done", output_index: outIdx, item })}\n\n`,
    fnCallArgsDelta: (outIdx, callId, delta) =>
      `event: response.function_call_arguments.delta\ndata: ${JSON.stringify({ type: "response.function_call_arguments.delta", output_index: outIdx, call_id: callId, delta })}\n\n`,
    fnCallArgsDone: (outIdx, callId, args) =>
      `event: response.function_call_arguments.done\ndata: ${JSON.stringify({ type: "response.function_call_arguments.done", output_index: outIdx, call_id: callId, arguments: args })}\n\n`,
    completed: (response) =>
      `event: response.completed\ndata: ${JSON.stringify({ type: "response.completed", response })}\n\n`,
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
  let thinkBuffer = "";
  let messageStarted = false;
  let completionSent = false;
  let lastDelta = ""; // dedup guard
  let recentSentences = []; // track last few sentences for dedup
  const toolCalls = new Map(); // index -> { id, name, arguments, outputIdx }
  let outputIndex = 0;
  let textOutputIdx = -1; // tracks where text was assigned during streaming
  let buffer = "";
  let streamOutput = null;

  const decoder = new TextDecoder();

  for await (const chunk of upstreamRes.body) {
    buffer += decoder.decode(chunk, { stream: true });

    const lines = buffer.split("\n");
    buffer = lines.pop(); // keep incomplete line

    for (const line of lines) {
      if (!line.startsWith("data: ")) continue;
      const data = line.slice(6).trim();
      if (data === "[DONE]") {
        // Stream ended — send completion if not already sent
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

      // Handle tool calls
      if (delta?.tool_calls) {
        for (const tc of delta.tool_calls) {
          const idx = tc.index ?? 0;
          // If text already started at index 0, offset tool calls by 1 to avoid index collision
          const tcOutIdx = (messageStarted && textOutputIdx === 0) ? outputIndex + idx + 1 : outputIndex + idx;
          if (!toolCalls.has(idx)) {
            const callId = tc.id || `call_${uid()}`;
            const fcId = `fc_${uid()}`;
            toolCalls.set(idx, { id: fcId, callId, name: tc.function?.name || "", arguments: "", outputIdx: tcOutIdx });
            const item = {
              type: "function_call",
              id: fcId,
              call_id: callId,
              name: tc.function?.name || "",
              arguments: "",
              status: "in_progress",
            };
            res.write(events.outputItemAdded(tcOutIdx, item));
          }
          if (tc.function?.arguments) {
            const tc_data = toolCalls.get(idx);
            tc_data.arguments += tc.function.arguments;
            res.write(events.fnCallArgsDelta(tc_data.outputIdx, tc_data.callId, tc.function.arguments));
          }
        }
        // Check finish_reason on same chunk before continuing (don't skip it!)
        if (finishReason && !completionSent) {
          completionSent = true;
          streamOutput = sendCompletion(res, events, responseId, model, fullText, toolCalls, outputIndex, textOutputIdx, finishReason, parsed.usage, previousResponseId, metadata);
        }
        continue;
      }

      // Skip reasoning_content (thinking) — only process content
      if (delta && delta.reasoning_content) continue;

      // Handle content — with reasoning_split=true, content should be clean (no <think> tags)
      if (delta && delta.content) {
        let text = delta.content;

        // Fallback: strip <think> tags if they still appear
        if (text.includes("<think>")) { inThink = true; text = text.replace(/<think>/g, ""); }
        if (text.includes("</think>")) { inThink = false; text = text.replace(/<\/think>/g, ""); }
        if (inThink) continue; // skip thinking content

        if (!text) continue;

        if (!messageStarted) {
          messageStarted = true;
          textOutputIdx = outputIndex + toolCalls.size;
          res.write(events.outputItemAdded(textOutputIdx, {
            type: "message", id: `msg_${uid()}`, status: "in_progress", role: "assistant", content: [],
          }));
          res.write(events.contentPartAdded(textOutputIdx, 0, { type: "output_text", text: "", annotations: [] }));
        }

        fullText += text;
        res.write(events.textDelta(textOutputIdx, 0, text));
      }

      // Check for finish
      if (finishReason && !completionSent) {
        completionSent = true;
        streamOutput = sendCompletion(res, events, responseId, model, fullText, toolCalls, outputIndex, textOutputIdx, finishReason, parsed.usage, previousResponseId, metadata);
      }
    }
  }

  // Fallback: if stream ended without finish_reason or [DONE]
  if (!completionSent) {
    completionSent = true;
    // If we had content or tool calls in progress, this was likely truncated — signal incomplete
    const wasGenerating = fullText.length > 0 || toolCalls.size > 0;
    const fallbackReason = wasGenerating ? "length" : "stop";
    console.warn(`[proxy] stream ended without finish_reason (wasGenerating=${wasGenerating}, reason=${fallbackReason})`);
    streamOutput = sendCompletion(res, events, responseId, model, fullText, toolCalls, outputIndex, textOutputIdx, fallbackReason, null, previousResponseId, metadata);
  }

  res.end();
  return { responseId, output: streamOutput || [] };
}

function sendCompletion(res, events, responseId, model, fullText, toolCalls, outputIndex, textOutputIdx, finishReason, usage, previousResponseId, metadata) {
  // Finalize tool calls using the output indices assigned during streaming
  for (const [idx, tc] of toolCalls) {
    const tcIdx = tc.outputIdx != null ? tc.outputIdx : outputIndex + idx;
    res.write(events.fnCallArgsDone(tcIdx, tc.callId, tc.arguments));
    const doneItem = {
      type: "function_call",
      id: tc.id,
      call_id: tc.callId,
      name: tc.name,
      arguments: tc.arguments,
      status: "completed",
    };
    res.write(events.outputItemDone(tcIdx, doneItem));
  }

  // Finalize text message — use the SAME index that was used during streaming
  const msgOutIdx = textOutputIdx >= 0 ? textOutputIdx : outputIndex + toolCalls.size;
  const trimmed = fullText.trim();
  if (trimmed) {
    const donePart = { type: "output_text", text: trimmed, annotations: [] };
    res.write(events.textDone(msgOutIdx, 0, trimmed));
    res.write(events.contentPartDone(msgOutIdx, 0, donePart));
    const doneMsg = {
      type: "message",
      id: `msg_${uid()}`,
      status: "completed",
      role: "assistant",
      content: [donePart],
    };
    res.write(events.outputItemDone(msgOutIdx, doneMsg));
  }

  // Build final output sorted by output index to match streaming order
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
  const finalOutput = outputItems.map(o => o.item);

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

// --- HTTP server ---

const server = http.createServer(async (req, res) => {
  // Health check
  if (req.method === "GET" && (req.url === "/health" || req.url === "/")) {
    res.writeHead(200, { "Content-Type": "application/json" });
    res.end(JSON.stringify({ status: "ok", proxy: "codex-minimax-proxy" }));
    return;
  }

  // Models endpoint
  if (req.method === "GET" && (req.url === "/v1/models" || req.url === "/models")) {
    res.writeHead(200, { "Content-Type": "application/json" });
    res.end(
      JSON.stringify({
        object: "list",
        data: [{ id: "MiniMax-M2.7", object: "model", owned_by: "minimax" }],
      })
    );
    return;
  }

  // Responses API -> Chat Completions
  if (req.method === "POST" && (req.url === "/v1/responses" || req.url === "/responses")) {
    let rawBody = "";
    for await (const chunk of req) rawBody += chunk;

    let body;
    try {
      body = JSON.parse(rawBody);
    } catch {
      res.writeHead(400, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ error: "Invalid JSON" }));
      return;
    }

    // Save original input for response store (before chain resolution)
    const originalInput = normalizeInputToArray(body.input);

    // Resolve previous_response_id chain — prepend stored context
    if (body.previous_response_id) {
      const chainItems = resolveResponseChain(body.previous_response_id);
      if (chainItems.length > 0) {
        const currentInput = normalizeInputToArray(body.input);
        body.input = [...chainItems, ...currentInput];
        console.log(`[proxy] resolved previous_response_id: ${chainItems.length} items prepended (total input: ${body.input.length})`);
      }
    }

    // Circuit breaker: detect tool-call loops and inject nudge
    if (body.previous_response_id) {
      const prevStored = responseStore.get(body.previous_response_id);
      const consecutiveTc = prevStored?.consecutiveToolCalls || 0;
      if (consecutiveTc >= MAX_CONSECUTIVE_TOOL_CALLS) {
        console.warn(`[proxy] CIRCUIT BREAKER: ${consecutiveTc} consecutive tool-call-only responses detected — injecting stop-loop nudge`);
        const nudge = {
          type: "message",
          role: "user",
          content: [{ type: "input_text", text: `[SYSTEM: You have made ${consecutiveTc} consecutive tool calls without responding to the user. You MUST now stop making tool calls and provide a text response summarizing your progress, findings, and any remaining work. Do NOT make any more tool calls in this response.]` }],
        };
        const currentInput = normalizeInputToArray(body.input);
        body.input = [...currentInput, nudge];
      } else if (consecutiveTc >= Math.floor(MAX_CONSECUTIVE_TOOL_CALLS * 0.75)) {
        // Soft warning at 75% threshold
        console.warn(`[proxy] tool-call loop warning: ${consecutiveTc}/${MAX_CONSECUTIVE_TOOL_CALLS} consecutive tool-call responses`);
      }
    }

    // Only route to OpenRouter if the ONLY tools are non-function (e.g. web_search alone)
    // If there are function tools too, use MiniMax and just strip web_search
    const hasWebSearch = body.tools?.some((t) => t.type === "web_search");
    const hasFunctionTools = body.tools?.some((t) => t.type === "function");
    const useOpenRouter = hasWebSearch && !hasFunctionTools && OPENROUTER_KEY;

    const chatReq = responsesRequestToChatCompletions(body);
    // Always override model to MiniMax-M2.7 (subagents may send different model names)
    chatReq.model = "MiniMax-M2.7";
    const isStream = chatReq.stream;

    // Route to OpenRouter for pure web search, MiniMax for everything else
    let upstreamUrl, upstreamKey, routeLabel;
    if (useOpenRouter) {
      chatReq.model = OPENROUTER_SEARCH_MODEL;
      // OpenRouter uses plugins for web search
      chatReq.plugins = [{ id: "web", max_results: 5 }];
      upstreamUrl = `${OPENROUTER_BASE}/chat/completions`;
      upstreamKey = OPENROUTER_KEY;
      routeLabel = `openrouter(${OPENROUTER_SEARCH_MODEL})`;
    } else {
      upstreamUrl = `${MINIMAX_BASE}/chat/completions`;
      upstreamKey = MINIMAX_KEY;
      routeLabel = "minimax";
    }

    // Hard circuit breaker: strip tools entirely to force text response
    if (body.previous_response_id) {
      const prevStored = responseStore.get(body.previous_response_id);
      const consecutiveTc = prevStored?.consecutiveToolCalls || 0;
      if (consecutiveTc >= MAX_CONSECUTIVE_TOOL_CALLS + 3) {
        console.warn(`[proxy] HARD CIRCUIT BREAKER: stripping all tools to force text response`);
        delete chatReq.tools;
        delete chatReq.tool_choice;
      }
    }

    console.log(`[proxy] ${routeLabel} | stream=${isStream} | messages=${chatReq.messages.length}${hasWebSearch ? " | web_search" : ""} | roles=[${chatReq.messages.map(m => m.role + (m.tool_calls ? "(tc)" : "")).join(",")}]`);

    // Add MiniMax-specific params when routing to MiniMax
    if (!useOpenRouter) {
      chatReq.reasoning_split = true; // separate thinking from content (no <think> tags)
    }

    try {
      const upstreamRes = await fetch(upstreamUrl, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${upstreamKey}`,
        },
        body: JSON.stringify(chatReq),
      });

      if (!upstreamRes.ok) {
        const errText = await upstreamRes.text();
        console.error(`[proxy] upstream error: ${upstreamRes.status} ${errText}`);
        if (!res.headersSent) {
          res.writeHead(upstreamRes.status, { "Content-Type": "application/json" });
          res.end(errText);
        }
        return;
      }

      if (isStream) {
        const { responseId: streamRespId, output: streamOutput } = await handleStreamingResponse(upstreamRes, res, body.model, body.previous_response_id, body.metadata);
        storeResponse(streamRespId, {
          input: originalInput,
          output: streamOutput,
          previousResponseId: body.previous_response_id || null,
        });
      } else {
        const ccResponse = await upstreamRes.json();
        const responsesResponse = chatCompletionToResponse(ccResponse, body.model, body.previous_response_id, body.metadata);
        storeResponse(responsesResponse.id, {
          input: originalInput,
          output: responsesResponse.output,
          previousResponseId: body.previous_response_id || null,
        });
        res.writeHead(200, { "Content-Type": "application/json" });
        res.end(JSON.stringify(responsesResponse));
      }
    } catch (err) {
      console.error(`[proxy] fetch error:`, err.message);
      if (!res.headersSent) {
        res.writeHead(502, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ error: { message: err.message } }));
      } else {
        res.end();
      }
    }
    return;
  }

  // Chat Completions — apply MiniMax fixups then forward
  if (req.method === "POST" && (req.url === "/v1/chat/completions" || req.url === "/chat/completions")) {
    let rawBody = "";
    for await (const chunk of req) rawBody += chunk;

    let body;
    try {
      body = JSON.parse(rawBody);
    } catch {
      res.writeHead(400, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ error: "Invalid JSON" }));
      return;
    }

    // Override model to MiniMax-M2.7
    body.model = "MiniMax-M2.7";
    const isStream = body.stream || false;

    // --- Apply MiniMax message fixups (same as /v1/responses path) ---
    let messages = body.messages || [];

    // Fix message ordering: tool results must follow their assistant(tc) message
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

    // Merge consecutive same-role messages
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
        // Drop text-only assistant after tool calls
      } else {
        merged.push(msg);
      }
    }

    // Validate: tool messages must follow assistant(tc) or another tool
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

    // Ensure tool_calls have valid JSON arguments string
    for (const msg of validated) {
      if (msg.role === "assistant" && msg.tool_calls) {
        for (const tc of msg.tool_calls) {
          if (!tc.function) continue;
          const args = tc.function.arguments;
          if (args === undefined || args === null || args === "") {
            tc.function.arguments = "{}";
            continue;
          }
          // If it's not a string, stringify it
          if (typeof args !== "string") {
            tc.function.arguments = JSON.stringify(args);
            continue;
          }
          // Validate it's actually valid JSON
          try {
            JSON.parse(args);
          } catch {
            console.warn(`[proxy] invalid tool_call arguments for ${tc.function.name} (id: ${tc.id}), wrapping as JSON: ${args.slice(0, 100)}`);
            // Try to salvage: wrap as a string value in an object
            tc.function.arguments = JSON.stringify({ input: args });
          }
        }
      }
    }

    // Also ensure tool result content is a string
    for (const msg of validated) {
      if (msg.role === "tool" && typeof msg.content !== "string") {
        msg.content = JSON.stringify(msg.content);
      }
    }

    body.messages = validated;

    // Add MiniMax-specific params
    body.reasoning_split = true;
    if (!body.max_tokens) body.max_tokens = 16384;

    console.log(`[proxy] chat/completions | stream=${isStream} | messages=${validated.length} | roles=[${validated.map(m => m.role + (m.tool_calls ? "(tc)" : "")).join(",")}]`);

    // Debug: log tool_calls details
    for (const msg of validated) {
      if (msg.role === "assistant" && msg.tool_calls) {
        for (const tc of msg.tool_calls) {
          console.log(`[proxy] tool_call: id=${tc.id} fn=${tc.function?.name} args=${(tc.function?.arguments || "").slice(0, 200)}`);
        }
      }
    }

    try {
      const upstreamRes = await fetch(`${MINIMAX_BASE}/chat/completions`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${MINIMAX_KEY}`,
        },
        body: JSON.stringify(body),
      });

      if (!upstreamRes.ok) {
        const errText = await upstreamRes.text();
        console.error(`[proxy] upstream error: ${upstreamRes.status} ${errText}`);
        if (!res.headersSent) {
          res.writeHead(upstreamRes.status, { "Content-Type": "application/json" });
          res.end(errText);
        }
        return;
      }

      if (isStream) {
        res.writeHead(200, {
          "Content-Type": "text/event-stream",
          "Cache-Control": "no-cache",
          Connection: "keep-alive",
        });
        for await (const chunk of upstreamRes.body) {
          res.write(chunk);
        }
        res.end();
      } else {
        const data = await upstreamRes.json();
        res.writeHead(200, { "Content-Type": "application/json" });
        res.end(JSON.stringify(data));
      }
    } catch (err) {
      console.error(`[proxy] fetch error:`, err.message);
      if (!res.headersSent) {
        res.writeHead(502, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ error: { message: err.message } }));
      } else {
        res.end();
      }
    }
    return;
  }

  // Pass through any other requests
  res.writeHead(404, { "Content-Type": "application/json" });
  res.end(JSON.stringify({ error: "Not found. Use POST /v1/responses" }));
});

// Increase timeouts for long-running streaming responses
server.timeout = 0; // no timeout
server.keepAliveTimeout = 300000; // 5 minutes
server.headersTimeout = 300000;
server.requestTimeout = 0;

server.listen(PORT, () => {
  console.log(`[codex-minimax-proxy] Listening on http://localhost:${PORT}`);
  console.log(`[codex-minimax-proxy] Primary: MiniMax @ ${MINIMAX_BASE}`);
  console.log(`[codex-minimax-proxy] Search:  ${OPENROUTER_KEY ? `OpenRouter (${OPENROUTER_SEARCH_MODEL})` : "DISABLED (no OPENROUTER_API_KEY)"}`);
});
