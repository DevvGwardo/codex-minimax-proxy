# codex-minimax-proxy

A lightweight proxy that translates OpenAI's **Responses API** (`/v1/responses`) into provider-native API formats, enabling [MiniMax M2.7](https://platform.minimax.io) and [Anthropic Claude](https://anthropic.com) to work with [OpenAI Codex CLI](https://github.com/openai/codex) and the Codex App.

## Why?

Codex CLI/App only supports the Responses API. MiniMax and Anthropic use different API formats. This proxy sits between them and translates on the fly.

```
Codex CLI/App  -->  /v1/responses  -->  [proxy]  -->  /v1/messages         -->  Anthropic API
                                                  -->  /v1/chat/completions -->  MiniMax API
```

## Features

- **Multi-provider support**: Route to Anthropic Claude or MiniMax based on model name
- Translates Responses API requests to each provider's native format
- Translates responses back to Responses API format
- Full streaming support (SSE translation for both providers)
- Function/tool calling translation
- Strips `<think>` reasoning tags from output
- MiniMax message ordering validation (tool results must follow tool calls)
- Context trimming for long conversations
- Optional web search routing via OpenRouter
- Circuit breaker for tool-call loops
- Zero dependencies — pure Node.js

## Claude Subscription Users (Pro/Max)

> **If you have a Claude Pro or Max subscription**, you don't need this proxy for Claude. Your subscription includes **[Claude Code CLI](https://docs.anthropic.com/en/docs/claude-code)** — an agentic coding tool with tool calling, file editing, terminal access, and sub-agents built in. Just run:
>
> ```bash
> claude
> ```
>
> This proxy's Anthropic integration is for users who want to use Claude models inside **OpenAI Codex CLI** specifically, which requires the **Anthropic API** (separate billing from your subscription).

## Quick Start

### 1. Get API Keys

- **Anthropic** (for Claude models): Sign up at [console.anthropic.com](https://console.anthropic.com)
- **MiniMax** (for MiniMax M2.7): Sign up at [platform.minimax.io](https://platform.minimax.io)
- **OpenRouter** (optional, for web search): Sign up at [openrouter.ai](https://openrouter.ai)

At least one of `ANTHROPIC_API_KEY` or `MINIMAX_API_KEY` is required.

### 2. Configure Environment

```bash
cp env.example .env
# Edit .env with your API keys
```

Or export directly:

```bash
export ANTHROPIC_API_KEY="your-key-here"
export MINIMAX_API_KEY="your-key-here"        # optional if using Anthropic
export OPENROUTER_API_KEY="your-key-here"      # optional
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

#### For Claude (Anthropic):

Edit `~/.codex/config.toml`:

```toml
model = "claude-sonnet-4-20250514"
model_provider = "anthropic-proxy"

[model_providers.anthropic-proxy]
name = "Anthropic (via proxy)"
base_url = "http://localhost:4000/v1"
env_key = "ANTHROPIC_API_KEY"
```

Available Claude models:
- `claude-opus-4-20250514` — Most capable
- `claude-sonnet-4-20250514` — Balanced (default)
- `claude-haiku-4-20250514` — Fast and efficient

#### For MiniMax:

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
| `ANTHROPIC_API_KEY` | (optional) | Your Anthropic API key |
| `ANTHROPIC_BASE_URL` | `https://api.anthropic.com` | Anthropic API base URL |
| `ANTHROPIC_MODEL` | `claude-sonnet-4-20250514` | Default Claude model |
| `MINIMAX_API_KEY` | (optional) | Your MiniMax API key |
| `MINIMAX_BASE_URL` | `https://api.minimax.io/v1` | MiniMax API base URL |
| `OPENROUTER_API_KEY` | (optional) | OpenRouter key for web search |
| `PROXY_PORT` | `4000` | Port the proxy listens on |
| `OPENROUTER_SEARCH_MODEL` | `nvidia/nemotron-3-super-120b-a12b:free` | Model for web search requests |

## Routing Logic

The proxy routes requests based on the **model name** in the request:

| Model Pattern | Route | Example |
|---|---|---|
| `claude*` or contains `anthropic` | Anthropic Messages API | `claude-sonnet-4-20250514` |
| Everything else | MiniMax Chat Completions API | `MiniMax-M2.7` |
| Web search only (no function tools) | OpenRouter | Automatic |

## How It Works

### Anthropic (Claude) Translation

| Responses API | Anthropic Messages API |
|---|---|
| `instructions` | `system` parameter |
| `input` (string/array) | `messages` array with content blocks |
| `input[].role: "developer"` | Appended to `system` prompt |
| `tools` (flat format) | Anthropic tool format with `input_schema` |
| `function_call` items | `tool_use` content blocks |
| `function_call_output` items | `tool_result` content blocks |
| `max_output_tokens` | `max_tokens` |

### MiniMax Translation

| Responses API | Chat Completions API |
|---|---|
| `instructions` | Prepended as user message |
| `input` (string) | `messages: [{ role: "user", content }]` |
| `input` (array) | Converted to messages array |
| `input[].role: "developer"` | Mapped to `"user"` role |
| `tools` (flat format) | Nested under `function` key |
| `max_output_tokens` | `max_tokens` |
| `reasoning.effort` | `reasoning_effort` |

### Response Translation (both providers → Responses API)

| Provider Response | Responses API |
|---|---|
| Text content | `output[].type: "message"` |
| Tool calls/use | `output[].type: "function_call"` |
| Token usage | `usage.input_tokens` / `output_tokens` |
| Stop/end_turn | `status: "completed"` |
| Max tokens | `status: "incomplete"` |

## Web Search

When Codex sends a request with `web_search` tools (and no function tools), the proxy routes it to an OpenRouter model with web search capability instead of MiniMax. This requires an `OPENROUTER_API_KEY`. (Web search routing is not used for Anthropic models.)

## Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Health check |
| `GET` | `/v1/models` | List available models |
| `POST` | `/v1/responses` | Responses API (translated to provider-native format) |
| `POST` | `/v1/chat/completions` | Chat Completions passthrough (MiniMax only) |

## Requirements

- Node.js 18+ (uses native `fetch`)
- At least one of: Anthropic API key, MiniMax API key

## License

MIT
