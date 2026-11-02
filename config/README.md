# Configuration

v0.1 needs no configuration file. Everything has a safe default, and the
defaults are what the test suite exercises. Configuration is read from
environment variables only, so there is no file here that could silently widen
the agent's reach.

## Environment variables

| Variable | Default | Effect |
| --- | --- | --- |
| `OMARCHY_AGENT_WORKSPACE_ROOTS` | `~/Projects` | `os.pathsep`-separated list of permitted filesystem roots. Entries that do not exist are dropped, so a typo cannot widen the policy. |
| `OMARCHY_AGENT_DATA_DIR` | `$XDG_DATA_HOME/omarchy-agent`, else `~/.local/share/omarchy-agent` | Where `agent.sqlite3` lives. |
| `XDG_DATA_HOME` | unset | Honoured when `OMARCHY_AGENT_DATA_DIR` is not set. |

Per-invocation overrides are available on the CLI instead:

```
python -m omarchy_agent --database /tmp/scratch.sqlite3 tasks
python -m omarchy_agent --endpoint http://127.0.0.1:8080/v1 models
python -m omarchy_agent files list ~/Projects/some-repo --root ~/Projects/some-repo
```

## What is deliberately not configurable

These are enforced in code, not in configuration, so that editing a file can
never relax them:

- **LocalAI must be on loopback.** `omarchy_agent/integrations/localai.py`
  rejects any non-loopback host before opening a socket.
- **No listener is ever bound.** The agent is a CLI; it opens no server port,
  and `network.bind_public` is a denied capability.
- **The capability table.** `omarchy_agent/security/permissions.py` is the one
  place that decides what is allowed, planned or forbidden. Adding a
  capability is a deliberate code change, reviewable in one diff.
- **Sensitive-file refusal.** The blocked-name and sensitive-name lists live in
  `omarchy_agent/security/paths.py`.

## Planned

A later milestone that adds command execution will introduce a
`config/allowlist.toml` holding the exact argument vectors the agent may run.
It does not exist yet, because command execution does not exist yet.
