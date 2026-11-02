# Omarchy Agent — Host Agent Core v0.1

A local, auditable automation foundation for an Omarchy Linux (Arch-based)
workstation. This is milestone **v0.1**: the agent can *look* at the machine
and keep its own notes. It cannot act on the machine yet.

> **What this version is not.** No AI model controls this computer. There is no
> command execution, no file modification, no desktop input, no browser
> automation and no network service. Those are later milestones, listed under
> [Roadmap](#roadmap). The permission table in
> `omarchy_agent/security/permissions.py` is the single source of truth for
> what is allowed, and the test suite asserts the planned capabilities refuse.

---

## Contents

- [Purpose](#purpose)
- [What works today](#what-works-today)
- [Architecture](#architecture)
- [Installation](#installation)
- [CLI commands](#cli-commands)
- [Database](#database)
- [Permission model](#permission-model)
- [Security limitations](#security-limitations)
- [LocalAI requirements](#localai-requirements)
- [Tests](#tests)
- [Troubleshooting](#troubleshooting)
- [Roadmap](#roadmap)

---

## Purpose

The long-term goal is an agent that can operate this workstation from natural
language, including from a phone over a private network. Reaching that safely
requires a foundation where every capability is explicit, every tool call is
audited, and the blast radius of a mistake is bounded by construction rather
than by good intentions.

v0.1 builds that foundation:

- a **path policy** that confines filesystem access to approved roots and
  refuses credential-shaped files;
- a **capability policy** in one module, so adding a power is a reviewable
  one-line change rather than a scattered edit;
- an **audit logger** that records every tool call with redacted arguments and
  never records file contents;
- **persistent storage** for tasks, memories, conversations, tool-execution
  metadata and approval requests;
- a **LocalAI client** that confirms the local inference server is up and reads
  its model list without guessing identifiers.

---

## What works today

| Area | Status | Notes |
| --- | --- | --- |
| Filesystem inspection | **Implemented** | Read-only. `list_directory`, `read_file`, `get_file_info`, `search_files`. |
| System information | **Implemented** | OS, CPU, memory, storage, GPU, process summary. |
| Git inspection | **Implemented** | Read-only status, branch, log, diff statistics. |
| LocalAI connectivity | **Implemented** | Health probe and model listing. |
| Persistent memory | **Implemented** | SQLite with schema versioning and migrations. |
| Audit logging | **Implemented** | In-memory ring buffer, JSONL sink, SQLite sink. |
| Permission policy | **Implemented** | Central table; allowed / planned / denied. |
| CLI | **Implemented** | Human-readable and `--json` output. |
| LocalAI inference | *Planned* | Interface fixed; refused by policy. |
| Terminal execution | *Planned* | Will be an argument-vector allowlist, never a shell string. |
| File modification | *Planned* | — |
| Desktop / browser control | *Planned* | — |
| MCP server | *Planned* | Bridges are already shaped as tools for this. |
| Remote access from a phone | *Planned* | Private network only. |

---

## Architecture

```
omarchy-agent/
├── omarchy_agent/
│   ├── __init__.py            package metadata and the scope statement
│   ├── __main__.py            `python -m omarchy_agent`
│   ├── cli.py                 argparse CLI, text and JSON renderers
│   │
│   ├── core/
│   │   └── task_manager.py    TaskManager + explicit initialize_storage()
│   │
│   ├── bridge/                the agent's read-only view of the host
│   │   ├── filesystem.py      list / read / info / search, policy-enforced
│   │   ├── system.py          /proc, /sys, shutil, optional nvidia-smi
│   │   └── git.py             fixed git subcommands, hooks neutralised
│   │
│   ├── memory/
│   │   └── database.py        SQLite schema, migrations, CRUD, scrubbing
│   │
│   ├── integrations/
│   │   └── localai.py         loopback-only OpenAI-shaped client
│   │
│   └── security/
│       ├── paths.py           PathPolicy: roots, blocked dirs, secret files
│       ├── permissions.py     the capability table
│       └── audit.py           AuditLogger and the @audited decorator
│
├── tests/                     283 unittest tests, fully hermetic
├── config/README.md           configuration surface (environment only)
├── pyproject.toml             metadata; zero runtime dependencies
└── README.md
```

### Design rules

1. **One gate per concern.** Path containment lives in `security/paths.py`;
   capability decisions live in `security/permissions.py`. Bridges call them;
   they do not re-implement them.
2. **Every tool is audited.** Bridge functions carry
   `@audited("tool.name", Capability.X)`, which performs the permission check,
   times the call, and records the outcome — including denials and errors.
3. **Results are dataclasses.** Each has `as_dict()` for JSON and
   `audit_summary()` for the log. The audit log sees the summary, never the
   payload, which is why reading a file cannot leak its contents into the log.
4. **Tool-shaped signatures.** Plain arguments, plain returns, no hidden state.
   Exposing these over MCP later is a wrapper, not a rewrite.
5. **Standard library only.** `sqlite3`, `urllib`, `subprocess`, `pathlib`,
   `argparse`, `dataclasses`, `unittest`. No runtime dependency to audit.

---

## Installation

Requires Python 3.12 or newer on Linux. Developed and tested on Python 3.15.2.

Nothing is installed globally, and no system package is touched.

### With `uv` (recommended)

```bash
cd ~/Projects/omarchy-agent
uv venv                       # creates ./.venv inside the project
./.venv/bin/python -m omarchy_agent --help
```

### With the standard library `venv`

```bash
cd ~/Projects/omarchy-agent
python3 -m venv .venv
./.venv/bin/python -m omarchy_agent --help
```

### No virtual environment at all

There are no runtime dependencies, so running from the project root also works:

```bash
cd ~/Projects/omarchy-agent
python3 -m omarchy_agent --help
```

### Optional: an `omarchy-agent` command

```bash
uv pip install -e .           # installs into ./.venv only
./.venv/bin/omarchy-agent --help
```

This is a convenience only; `python -m omarchy_agent` needs no install step.

---

## CLI commands

Every command accepts `--json` for machine-readable output, `--database PATH`
to point at a different store, `--endpoint URL` for a different loopback
LocalAI, `--show-audit` to print the run's audit entries, and `--persist-audit`
to append them to the database as `tool_executions` rows. `tasks` and `memory`
persist their own audit entries automatically, because they already hold a
database handle.

| Command | What it does | Writes anything? |
| --- | --- | --- |
| `audit` | Safe environment facts and tool availability. | No |
| `doctor` | Checks structure, policy, database and LocalAI; non-zero exit on failure. | No |
| `init` | Creates the data directory and database. Prints what it will create first. | **Yes**, only as described |
| `system` | OS, CPU, memory, storage, GPU and process summary. | No |
| `models` | LocalAI connection status and installed models. | No |
| `capabilities` | The full permission table. | No |
| `tasks` | List tasks; `--create` / `--complete` to modify. | Only with those flags |
| `memory` | List, filter or search memories; `--remember` to store. | Only with that flag |
| `files` | `list` / `read` / `info` / `search` inside permitted roots. | No |
| `git` | `status` / `branch` / `log` / `diff` on a repository. | No |
| `audit-log` | Tool-execution metadata persisted in the database. | No |

### Examples

```bash
python -m omarchy_agent doctor
python -m omarchy_agent init
python -m omarchy_agent system
python -m omarchy_agent models
python -m omarchy_agent capabilities

python -m omarchy_agent files list ~/Projects/omarchy-agent
python -m omarchy_agent files read ~/Projects/omarchy-agent/pyproject.toml
python -m omarchy_agent files search ~/Projects --query "nvidia-smi" --limit 20

python -m omarchy_agent git status ~/Projects/omarchy-agent
python -m omarchy_agent git log ~/Projects/omarchy-agent --limit 5

python -m omarchy_agent tasks --create "review the GPU report"
python -m omarchy_agent tasks
python -m omarchy_agent memory --remember facts gpu "RTX 4050, 6 GB VRAM" --tags hardware
python -m omarchy_agent memory --category facts

python -m omarchy_agent --json system | jq '.gpu.devices[0]'
python -m omarchy_agent --show-audit files read ~/Projects/omarchy-agent/README.md
python -m omarchy_agent --persist-audit system
python -m omarchy_agent audit-log --limit 10
```

### Exit codes

| Code | Meaning |
| --- | --- |
| `0` | Success |
| `1` | An operation failed or was refused |
| `2` | The database is not initialized — run `init` |
| `3` | A check failed (`doctor` errors, LocalAI unreachable) |
| `130` | Interrupted |

---

## Database

Default location:

```
~/.local/share/omarchy-agent/agent.sqlite3
```

Override with `OMARCHY_AGENT_DATA_DIR`, `XDG_DATA_HOME`, or `--database PATH`.

The directory and file are **only** created by `python -m omarchy_agent init`.
No other command, import or test creates anything outside the project tree.

### Schema (version 1)

| Table | Columns |
| --- | --- |
| `conversations` | `id`, `title`, `created_at`, `updated_at` |
| `messages` | `id`, `conversation_id` → `conversations`, `role`, `content`, `created_at` |
| `tasks` | `id`, `title`, `description`, `status`, `created_at`, `updated_at`, `completed_at` |
| `memories` | `id`, `category`, `key`, `content`, `tags`, `created_at`, `updated_at` — unique on `(category, key)` |
| `tool_executions` | `id`, `task_id` → `tasks`, `tool_name`, `arguments`, `result_summary`, `status`, `created_at`, `duration_ms` |
| `approvals` | `id`, `action`, `description`, `status`, `created_at`, `resolved_at` |

### Guarantees

- **Non-destructive initialization.** An existing database is opened and
  migrated forward. `init` never drops, truncates or replaces a file. Running
  it twice is safe and is covered by a test.
- **Schema versioning.** `PRAGMA user_version` plus an ordered, additive
  migration list. A database written by a *newer* build is refused rather than
  downgraded.
- **Foreign keys enforced** (`PRAGMA foreign_keys = ON`). Deleting a
  conversation cascades to its messages; deleting a task nulls the link from
  its tool-execution rows.
- **Explicit transactions.** `BEGIN IMMEDIATE` … `COMMIT`, rolling back on any
  exception.
- **Parameterized SQL only.** No value is ever interpolated into a statement.
  SQL-injection resistance is covered by six tests.
- **Busy timeout** of 5 s, WAL journal mode.
- **UTC timestamps** in ISO-8601, second precision.
- **Private permissions.** Directory `0700`, database file `0600`.
- **Secret scrubbing before storage.** Token- and key-shaped text and
  `SECRET_NAME=value` lines are replaced with `[REDACTED]` on the way in, so a
  careless `remember()` cannot persist a credential.
- **Length caps.** 16 KiB per content field, 1 KiB per result summary, so large
  outputs are never retained wholesale.

---

## Permission model

`omarchy_agent/security/permissions.py` holds one table mapping every
capability to exactly one of three states. `python -m omarchy_agent
capabilities` prints it.

### Allowed in v0.1

`filesystem.read`, `filesystem.list`, `filesystem.search`, `system.info`,
`git.inspect`, `localai.inspect`, `memory.read`, `memory.write`, `task.read`,
`task.write`, `audit.write`.

### Planned — not implemented (raises `CapabilityNotImplemented`)

`filesystem.write`, `filesystem.delete`, `terminal.execute`,
`localai.inference`, `browser.control`, `desktop.input`,
`desktop.screenshot`, `application.manage`, `schedule.write`,
`remote.access`.

### Denied by project policy (raises `PermissionDenied`, now and later)

`system.privileged` (no `sudo`), `system.package_install` (no `pacman`, no
global installs), `docker.control` (never start, stop or reconfigure the
LocalAI container; never touch `/var/run/docker.sock`),
`system.service_install` (no systemd unit, cron job or daemon),
`secret.read` (no SSH keys, credential stores or password databases),
`network.bind_public` (never bind `0.0.0.0`).

An unknown capability fails closed.

### Audit record

Every bridge call records tool name, UTC timestamp, redacted arguments,
`success` / `error` / `denied`, a result *summary*, duration in milliseconds,
and the capability involved. File contents, secrets and private keys are never
recorded: argument names matching `password|secret|token|api_key|…` are
replaced, credential-shaped *values* are caught by pattern regardless of the
name, and results are summarized by shape (`read … 412 bytes, 18 lines`) rather
than by content.

Sinks: the in-memory ring buffer (always), a `0600` JSONL file, and the
`tool_executions` table.

### Filesystem policy

- Default permitted root: `~/Projects`. Override with
  `OMARCHY_AGENT_WORKSPACE_ROOTS` (`os.pathsep`-separated); entries that do not
  exist are dropped, so a typo cannot widen the policy.
- Paths are **fully resolved before** any containment check, which is what
  defeats `../` traversal and symlink escapes. A symlink pointing outside the
  roots is refused, including a symlinked *directory* used as a path prefix.
- Skipped directories: `.git`, `node_modules`, `.venv`, `venv`,
  `__pycache__`, `build`, `dist`, `target`, `.next`, `.cache`, `*.egg-info` and
  more — plus credential stores (`.ssh`, `.gnupg`, `.aws`, `.password-store`,
  `.mozilla`, `.docker`) even if symlinked in.
- Refused files: `.env` and friends, `id_rsa`, `id_ed25519`, `*.pem`, `*.key`,
  `.netrc`, `.pgpass`, `.npmrc`, `.git-credentials`, `credentials.json`,
  `*.kdbx`, browser login databases, and similar.
- Allowed anyway: `.env.example`, `.env.sample`, `.env.template`, `*.example`.
- 128 KiB maximum per file read.
- Binary files are detected (NUL bytes, invalid UTF-8, control-byte density)
  and refused rather than mangled into text.
- Searches are bounded by result count, wall-clock timeout and file count; the
  query is a literal case-insensitive substring, never a regex or glob.

### Subprocess policy

Two modules run external programs: `bridge/system.py` (only `nvidia-smi`) and
`bridge/git.py` (only `rev-parse`, `status`, `log`, `diff`, `symbolic-ref`).
Both use a fixed argument array, an explicit timeout and a minimal environment.
`shell=True` appears nowhere in the project, and a test asserts no `shell`
keyword is passed. Git runs with `core.hooksPath=`, `core.fsmonitor=false`,
`--no-optional-locks`, `GIT_CONFIG_NOSYSTEM=1`, `GIT_TERMINAL_PROMPT=0` and an
empty `GIT_ASKPASS`, so no repository hook, monitor or credential helper
executes; a test plants a hook and asserts it does not run.

---

## Security limitations

Honest boundaries of this version:

1. **This is not a sandbox.** The agent runs as your user and inherits your
   permissions. The policy is a careful gate, not a kernel boundary. Nothing
   here would stop a bug in Python itself, and the code is not hardened against
   a local attacker who can already edit these files.
2. **Metadata is still visible.** `get_file_info` reports that `.env` exists
   and how large it is. It never opens it, but the existence is observable.
3. **Search results contain file content.** That is the point of a search.
   Credential-shaped matches are redacted and sensitive files are never opened,
   but a match line from an ordinary file is returned verbatim to the caller
   (never to the audit log).
4. **Secret detection is heuristic.** The name and shape lists catch common
   cases. A secret in an unusually named file with an unusual shape can still be
   read. Keep real secrets out of `~/Projects`.
5. **Scrubbing is best-effort.** `remember()` scrubs on the way in, but a
   credential in an unrecognised format could be persisted. Do not feed raw
   credential material to the agent.
6. **Process names are reported.** Command lines and environment blocks are
   deliberately not collected, but a process *name* can hint at what is
   running.
7. **No authentication, no encryption at rest.** The database is a local file
   protected by `0600`. Anyone who can read your home directory can read it.
8. **TOCTOU.** Path checks and the subsequent open are separate syscalls. A
   local attacker who can swap a path between the two could in principle defeat
   the check. The agent is not designed for a hostile local user.
9. **No rate limiting or resource quota** beyond the per-call timeouts and
   limits described above.

---

## LocalAI requirements

- Endpoint: `http://127.0.0.1:8080/v1` (override with `--endpoint`).
- **Loopback only.** `127.0.0.1`, `::1` and `localhost` are accepted; any other
  host is rejected before a socket is opened. This is enforced in code, not in
  configuration.
- Connect timeout 2 s, read timeout 5 s. A 1 MiB response cap.
- The agent **never** starts, stops, restarts or reconfigures Docker, the
  `local-ai` container, its volumes, or the NVIDIA runtime. It makes one HTTP
  GET. If the server is down, `models` and `doctor` say so and exit non-zero.
- **Model identifiers are never assumed.** `list_models()` reads them from the
  server; `resolve_model_id()` refuses a name the server does not report. An
  empty model list is a valid, expected state — it means the gallery install
  has not finished.
- Inference is not implemented. `chat_completion()` exists to fix the interface
  and refuses via the permission policy.

---

## Tests

```bash
cd ~/Projects/omarchy-agent

# everything
./.venv/bin/python -m unittest discover -s tests -t .

# verbose
./.venv/bin/python -m unittest discover -s tests -t . -v

# one module
./.venv/bin/python -m unittest tests.test_security
./.venv/bin/python -m unittest tests.test_filesystem

# one test
./.venv/bin/python -m unittest tests.test_filesystem.ReadFileTests.test_rejects_binary_file
```

283 tests across seven modules. Properties of the suite:

- **No network.** HTTP responses are faked at the URL-opener level.
- **No real data touched.** Temporary directories and temporary SQLite files
  only; `~/Projects` and `~/.local/share/omarchy-agent` are never read or
  written.
- **External commands mocked**, except a handful that run the real `git`,
  `true` and `false`. The `nvidia-smi` test skips itself when that binary is
  absent.

Coverage of the security-relevant behaviour includes: `../` traversal, symlink
escape (file and directory), NUL bytes in paths, blocked directories,
sensitive-file refusal, example-file allowance, oversized-file rejection,
binary-file rejection, search bounds, literal-not-regex queries, permission
denial for planned and forbidden capabilities, audit redaction by name and by
shape, "file content never reaches the log", SQL injection in keys, titles,
search strings, status filters and tool names, foreign-key enforcement,
transaction rollback, non-destructive re-initialization, file permissions, git
timeout and failure handling, git hooks not executing, LocalAI success,
failure, empty model list and malformed bodies, and the refusal of
non-loopback endpoints.

---

## Troubleshooting

**`error: database not found … run 'python -m omarchy_agent init' first`**
Expected before first use. Run `init`. It prints what it will create.

**`doctor` reports `workspace roots: no permitted root resolved`**
`~/Projects` does not exist. Create it, or set
`OMARCHY_AGENT_WORKSPACE_ROOTS=/path/to/code`.

**`PathSecurityError: path is outside the permitted workspace roots`**
Working by design. Move the target under `~/Projects`, or pass
`--root /that/directory` to the `files` command for a one-off widening.

**`PathSecurityError: refusing to access a file that may contain secrets`**
The file name matches the sensitive list. Use `files info` for metadata, or
rename to an `*.example` variant if it genuinely holds no secret.

**`BinaryFileRejected` / `FileTooLarge`**
By design: 128 KiB and UTF-8 text only. `files info` still reports metadata.

**`models` exits 3 with "cannot reach LocalAI"**
Check the container yourself — the agent will not touch Docker:
```bash
docker ps --filter name=local-ai
curl -s http://127.0.0.1:8080/v1/models
```

**`models` works but lists nothing**
The API is up and no model is installed yet. Finish the gallery installation.
`doctor` reports this as a warning, not an error.

**`GpuInformation(available=False)`**
`nvidia-smi` is missing, timed out, or reported a driver mismatch; `detail`
says which. The agent will not install GPU tooling.

**`GitError: … timed out`**
A very large repository, or a slow filesystem. Raise the timeout in the Python
API (`timeout=`); the CLI uses the 10 s default.

**`sqlite3.OperationalError: database is locked`**
Another process holds a write lock. The busy timeout is 5 s; retry.

**`DatabaseError: database schema version N is newer than this build supports`**
The file was written by a newer version of the agent. Upgrade rather than
downgrade; the agent refuses to touch it.

---

## Roadmap

Each step is a permission-table change plus an implementation, in this order:

1. **LocalAI inference and tool calling** — `localai.inference`, driving the
   existing bridges as tools.
2. **Allowlisted command execution** — `terminal.execute` as a table of
   argument vectors, with an approval record for anything outside it. Never a
   model-supplied shell string.
3. **Scoped file modification** — `filesystem.write` with diff previews and
   approvals.
4. **MCP server** — exposing the bridges unchanged over MCP.
5. **Desktop control** — `desktop.screenshot` then `desktop.input` via
   Hyprland and Wayland tooling.
6. **Browser automation** — `browser.control`.
7. **Scheduled workflows** — `schedule.write`, user-initiated only.
8. **Remote access from a phone** — `remote.access`, over a private network,
   authenticated, never bound to `0.0.0.0`.

The forbidden set does not move.

---

## License

MIT.
