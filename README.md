# codex-minimax-proxy

A lightweight proxy that translates OpenAI's **Responses API** (`/v1/responses`) into **Chat Completions API** (`/v1/chat/completions`), enabling [MiniMax M2.7](https://platform.minimax.io) to work with [OpenAI Codex CLI](https://github.com/openai/codex) and the Codex App.

## Why?

Codex CLI/App only supports the Responses API. MiniMax M2.7 only supports Chat Completions. This proxy sits between them and translates on the fly.

```
Codex CLI/App  -->  /v1/responses  -->  [proxy]  -->  /v1/chat/completions  -->  MiniMax API
```

## Features

- Translates Responses API requests to Chat Completions format
- Translates Chat Completions responses back to Responses API format
- Full streaming support (SSE translation)
- Function/tool calling translation (flat ↔ nested format)
- Strips `<think>` reasoning tags from output
- MiniMax message ordering validation (tool results must follow tool calls)
- Context trimming for long conversations
- Optional web search routing via OpenRouter
- Zero dependencies — pure Node.js

## Quick Start

### 1. Get API Keys

- **MiniMax** (required): Sign up at [platform.minimax.io](https://platform.minimax.io)
- **OpenRouter** (optional, for web search): Sign up at [openrouter.ai](https://openrouter.ai)

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env with your API keys
```

Or export directly:

```bash
export MINIMAX_API_KEY="your-key-here"
export OPENROUTER_API_KEY="your-key-here"  # optional
```

### 3. Start the Proxy

```bash
node proxy.mjs
```

Or run in the background:

```bash
nohup node proxy.mjs > /tmp/codex-minimax-proxy.log 2>&1 &
```

### 4. Configure Codex CLI

Edit `~/.codex/config.toml`:

```toml
model = "MiniMax-M2.7"
model_provider = "minimax"

[model_providers.minimax]
name = "MiniMax"
base_url = "http://localhost:4000/v1"
env_key = "MINIMAX_API_KEY"
```

### 5. Run Codex

```bash
codex
```

## Configuration

| Environment Variable | Default | Description |
|---|---|---|
| `MINIMAX_API_KEY` | (required) | Your MiniMax API key |
| `OPENROUTER_API_KEY` | (optional) | OpenRouter key for web search |
| `PROXY_PORT` | `4000` | Port the proxy listens on |
| `MINIMAX_BASE_URL` | `https://api.minimax.io/v1` | MiniMax API base URL |
| `OPENROUTER_SEARCH_MODEL` | `nvidia/nemotron-3-super-120b-a12b:free` | Model for web search requests |

## Web Search

When Codex sends a request with `web_search` tools (and no function tools), the proxy routes it to an OpenRouter model with web search capability instead of MiniMax. This requires an `OPENROUTER_API_KEY`.

Regular coding requests always go to MiniMax.

## How It Works

### Request Translation (Responses → Chat Completions)

| Responses API | Chat Completions API |
|---|---|
| `instructions` | Prepended as user message |
| `input` (string) | `messages: [{ role: "user", content }]` |
| `input` (array) | Converted to messages array |
| `input[].role: "developer"` | Mapped to `"user"` role |
| `tools` (flat format) | Nested under `function` key |
| `max_output_tokens` | `max_tokens` |
| `reasoning.effort` | `reasoning_effort` |

### Response Translation (Chat Completions → Responses)

| Chat Completions | Responses API |
|---|---|
| `choices[0].message.content` | `output[].type: "message"` |
| `choices[0].message.tool_calls` | `output[].type: "function_call"` |
| `usage.prompt_tokens` | `usage.input_tokens` |
| `finish_reason: "stop"` | `status: "completed"` |

### MiniMax-Specific Handling

- **No system role**: `system`/`developer` messages converted to `user` messages
- **Tool ordering**: Tool results must directly follow their tool call — the proxy validates and fixes this
- **Reasoning split**: Sends `reasoning_split: true` to separate thinking from content
- **Context trimming**: Keeps conversations under 40 messages to prevent timeouts

## Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Health check |
| `GET` | `/v1/models` | List available models |
| `POST` | `/v1/responses` | Responses API (translated to Chat Completions) |

## Requirements

- Node.js 18+ (uses native `fetch`)
- MiniMax API key

## License

MIT
