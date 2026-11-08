# Omarchy Agent

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python: >=3.12](https://img.shields.io/badge/python-3.12%2B-blue.svg)](pyproject.toml)
[![Dependencies](https://img.shields.io/badge/dependencies-rich-brightgreen.svg)](pyproject.toml)

**Omarchy Agent** is a secure, auditable, local-first host automation foundation and AI assistant CLI designed for Linux workstations. Backed by local inference (LocalAI) and bounded by a strict capability security model, Omarchy Agent enables repository exploration, system diagnostics, task tracking, and interactive AI chat without exposing credentials or executing unvetted commands.

---

## Key Features

- **Interactive Local AI Chat (`loc`)**: Context-aware terminal chat interface powered by local inference models (e.g. Qwen, Llama via LocalAI). Features streaming output, markdown formatting, turn management, tool calling (read/write/edit files, shell execution with human confirmation), and `/plan` goal decomposition.
- **Auditable Host & Workspace Bridges**: Hardened bridges for filesystem read/write/edit, command execution in workspace roots, Git status and history, hardware telemetry (CPU, GPU, memory, disks), and running processes.
- **Remote Access Server (`serve`)**: Lightweight standard library HTTP/SSE streaming server with Bearer token authentication, ready for subdomain reverse proxies (e.g., Caddy, Nginx, Cloudflare Tunnel).
- **Security by Construction**: Strict workspace path policy (`PathPolicy`) that prevents directory traversal, symlink escapes, and access to sensitive/credential files (`.env`, SSH keys, `.git`, certificates). Commands require interactive user approval.
- **Immutable Permission Model**: Centralized capability matrix (`CapabilityState`: `ALLOWED`, `NOT_IMPLEMENTED`, `DENIED`) enforcing non-negotiable boundaries (no privilege escalation, no shell-injection paths, no public network binding).
- **Persistent Memory & Task Engine**: Structured SQLite storage with schema versioning, WAL mode, transaction isolation, and automatic redaction of sensitive credentials.
- **Lightweight & Modular**: Built on Python standard library foundations (`sqlite3`, `urllib`, `subprocess`, `pathlib`, `http.server`) with `rich` for terminal UI presentation. Zero added external dependencies.

---

## Architecture Overview

```
omarchy-agent/
├── omarchy_agent/
│   ├── cli.py                  # Command-line interface and dispatchers
│   ├── core/
│   │   ├── chat.py             # Chat session orchestrator, tool dispatch, /plan
│   │   ├── terminal_ui.py      # Terminal UI renderer and activity indicators
│   │   └── task_manager.py     # High-level task and storage controller
│   ├── bridge/
│   │   ├── filesystem.py       # Safe filesystem operations (read, write, edit)
│   │   ├── terminal.py         # Audited shell command execution with timeout
│   │   ├── git.py              # Safe Git inspection bridge (hardened argv)
│   │   └── system.py           # System, hardware, and OS telemetry
│   ├── integrations/
│   │   └── localai.py          # Loopback-only LocalAI client and streaming engine
│   ├── memory/
│   │   └── database.py         # SQLite database schema, migrations, and CRUD
│   ├── remote/
│   │   └── server.py           # Token-authenticated HTTP/SSE remote server
│   └── security/
│       ├── paths.py            # Path containment, symlink checking, secret refusal
│       ├── permissions.py      # Central capability and permission policies
│       └── audit.py            # Redacting audit logger and function decorator
├── tests/                      # Hermetic unit test suite (330 tests)
└── pyproject.toml              # Build configuration and project metadata
```

---

## Installation & Setup

Omarchy Agent requires **Python 3.12 or newer** on Linux.

### Recommended: Using `uv`

```bash
# Clone the repository
git clone https://github.com/rithvin-us/omarchy-agent.git
cd omarchy-agent

# Create virtual environment and install in editable mode
uv venv
uv pip install -e .
```

### Using Standard Python `venv`

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

---

## Quick Start

### 1. Run System Doctor

Verify environment integrity, database configuration, workspace containment, and LocalAI connectivity:

```bash
omarchy-agent doctor
```

### 2. Initialize Persistent Storage

Initialize the local SQLite database at `~/.local/share/omarchy-agent/agent.sqlite3` (non-destructive):

```bash
omarchy-agent init
```

### 3. Launch Interactive AI Chat

Start an interactive conversation with the local model about your codebase:

```bash
# Using the short alias
loc

# Or using the full command line with specific model and root options
omarchy-agent chat --root ~/Projects/omarchy-agent --model qwen3.8-4b-q4
```

In-chat commands:
- `/help`: Display chat instructions and available tool capabilities.
- `/files`: List files utilized as context in the previous turn.
- `/plan <goal>`: Decompose a goal into numbered steps and record tasks in the SQLite database.
- `/clear`: Clear message history and reset context window.
- `/exit`: Terminate session.

### 4. Remote Access Server

Start the token-authenticated remote server (ideal for reverse proxies such as Caddy or Cloudflare Tunnel):

```bash
# Start server on 127.0.0.1:8443 with a custom token
omarchy-agent serve --port 8443 --token my-secret-token

# Available endpoints:
# GET  /api/health - Health check, model info, and workspace root
# GET  /api/tasks  - List all tasks stored in the database
# POST /api/chat   - Submit prompts with optional SSE streaming (stream: true)
```

For full reverse proxy configurations (Cloudflare Tunnel, Caddy, Nginx), SSE streaming clients, and systemd service setup, see the [Remote Access Guide](docs/REMOTE_GUIDE.md).

---

## Command Reference

| Command | Description | Write Action |
| :--- | :--- | :---: |
| `doctor` | Verify workspace permissions, database integrity, and LocalAI status | No |
| `audit` | Display safe host metadata, Python runtime, and tool availability | No |
| `system` | Inspect CPU, memory, filesystem volumes, GPU (via `nvidia-smi`), and processes | No |
| `models` | List installed LocalAI models and endpoint latency | No |
| `capabilities` | Display the central capability state matrix | No |
| `chat` | Start interactive chat with tool calling and `/plan` (`loc`) | File/Cmd (Prompted) |
| `serve` | Run token-authenticated HTTP/SSE server for remote access | No |
| `files` | Bounded workspace search and inspection (`list`, `read`, `info`, `search`) | No |
| `git` | Safe repository inspection (`status`, `branch`, `log`, `diff`) | No |
| `tasks` | View, add, or transition task statuses (`--create`, `--complete`) | Database |
| `memory` | View, search, or record persistent agent notes (`--remember`) | Database |
| `audit-log` | Review historical tool execution audit entries from the database | No |

### Global Flags

- `--json`: Output machine-readable JSON for integration into scripts or tools.
- `--database <path>`: Override the default SQLite database path.
- `--endpoint <url>`: Override the LocalAI base URL (loopback only: `127.0.0.1`, `localhost`, `::1`).
- `--show-audit`: Print real-time in-memory audit log upon command completion.
- `--persist-audit`: Persist execution audit entries to the database.

---

## Security Model

Omarchy Agent is built around explicit security guarantees:

1. **Path Isolation**: Filesystem access is strictly confined to designated workspace roots (defaults to `~/Projects`, configurable via `OMARCHY_AGENT_WORKSPACE_ROOTS`). Paths are resolved before validation, preventing symlink bypasses or `../` directory traversals.
2. **Credential Protection**: Known sensitive filenames (`.env*`, `id_rsa`, `*.pem`, `*.key`, `*.kdbx`, `credentials.json`) are blocked by policy. All audit arguments and database memory writes are scrubbed of credential patterns before persistence.
3. **Loopback Isolation**: The LocalAI client communicates exclusively over local loopback interfaces (`127.0.0.1`, `::1`, `localhost`). External hosts are refused at validation.
4. **Hardened Command Execution**: Process calls avoid shell invocation (`shell=False`) and use static, immutable argument vectors. Git bridge calls enforce `-c core.hooksPath=` and neutralised environments to prevent execution of unvetted hooks or helpers.
5. **Private Data Storage**: SQLite database files are initialized with `0600` permissions and directories with `0700`.

---

## Development & Testing

### Running Tests

The test suite is fully hermetic: external commands are isolated, HTTP requests are mocked, and temporary directories are utilized.

```bash
# Run all tests
python -m unittest discover -s tests

# Run specific test modules
python -m unittest tests.test_chat
python -m unittest tests.test_security
```

### Code Quality & Formatting

```bash
# Lint checks
ruff check .

# Code formatting
ruff format --check .
```

---

## License

This project is licensed under the [MIT License](LICENSE).


## Local voice, TUI, Google, memory

```bash
uv sync --extra voice --extra tui          # Python 3.13 (see .python-version)
uv run omarchy-agent tui                   # F2 talk, ctrl+s speak replies, y/n approvals
uv run omarchy-agent voice                 # hands-free; spoken "yes" approves
uv run omarchy-agent google-login          # Gmail (read) + Calendar; needs
                                           # ~/.config/omarchy-agent/google_client.json
uv run omarchy-agent serve                 # web client + API; see docs/REMOTE_GUIDE.md
```

Speech-to-text (faster-whisper `small.en`) and text-to-speech (Piper) run locally; models
live in `~/.local/share/omarchy-agent/`. Tools are risk-tiered (`security/risk.py`):
GREEN/YELLOW run freely, ORANGE/RED always ask. Long-term memory tools: `remember`,
`recall`, `forget`.
