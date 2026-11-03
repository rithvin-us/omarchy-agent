# Omarchy Agent Remote Control Guide

This guide explains how to configure, secure, and access your **Omarchy Agent** remotely from a subdomain, mobile device, or external network using the built-in HTTP/SSE server and a reverse proxy.

---

## 1. Overview & Architecture

`omarchy-agent serve` provides an authenticated, lightweight HTTP and Server-Sent Events (SSE) streaming API. By default, it binds to `127.0.0.1:8443` and requires a Bearer token for every request.

```
┌─────────────────────────────────┐
│ Client (Phone, Browser, Script) │
└────────────────┬────────────────┘
                 │ HTTPS (Bearer Token)
                 ▼
┌─────────────────────────────────┐
│ Reverse Proxy (Subdomain)       │
│ Caddy / Cloudflare / Nginx      │
└────────────────┬────────────────┘
                 │ Loopback (http://127.0.0.1:8443)
                 ▼
┌─────────────────────────────────┐
│ omarchy-agent serve             │
│ (ChatSession, Tools, Database)  │
└─────────────────────────────────┘
```

### Security Highlights
- **Strict Bearer Authentication**: Uses constant-time token comparison (`hmac.compare_digest`) to prevent timing side-channel attacks.
- **Loopback Default**: Never binds publicly (`0.0.0.0` is blocked by central capability policy `Capability.NETWORK_BIND_PUBLIC`).
- **Audited**: Server instantiation is logged under `Capability.REMOTE_ACCESS`.
- **Path Containment**: File read, write, and edit operations remain strictly bound to the workspace root (`PathPolicy`).

---

## 2. Starting the Server

Activate the virtual environment and start the server:

```bash
source .venv/bin/activate

# Start on 127.0.0.1:8443 with a chosen token
omarchy-agent serve --port 8443 --token "my-secure-remote-token"
```

If `--token` is omitted, `omarchy-agent serve` automatically generates a secure, high-entropy 24-byte random URL-safe token and prints it to the console.

### CLI Options

| Flag | Default | Description |
| :--- | :--- | :--- |
| `--host` | `127.0.0.1` | Loopback address to bind. |
| `--port` | `8443` | Port to listen on. |
| `--token` | *(auto-generated)* | Bearer authentication token. |
| `--root` | *(current directory)* | Workspace directory for file and git operations. |
| `--endpoint`| `http://127.0.0.1:8080/v1` | LocalAI API endpoint. |

---

## 3. Reverse Proxy & Subdomain Setup

To securely expose `127.0.0.1:8443` under a domain like `agent.example.com`, use any of the options below.

### Option A: Cloudflare Tunnel (Recommended — No Port Forwarding)

Cloudflare Tunnels let you expose local services to a subdomain without opening router firewall ports or configuring dynamic DNS:

1. Install `cloudflared`:
   ```bash
   sudo pacman -S cloudflared   # or your distro package manager
   ```
2. Authenticate and create a tunnel:
   ```bash
   cloudflared tunnel login
   cloudflared tunnel create omarchy-tunnel
   ```
3. Add an ingress rule in `~/.cloudflared/config.yml`:
   ```yaml
   tunnel: <TUNNEL_UUID>
   credentials-file: /home/user/.cloudflared/<TUNNEL_UUID>.json

   ingress:
     - hostname: agent.example.com
       service: http://127.0.0.1:8443
     - service: http_status:404
   ```
4. Route DNS and start the tunnel:
   ```bash
   cloudflared tunnel route dns omarchy-tunnel agent.example.com
   cloudflared tunnel run omarchy-tunnel
   ```

---

### Option B: Caddy (Automatic HTTPS)

Caddy automatically handles Let's Encrypt certificates and supports SSE streaming out of the box:

```caddy
# /etc/caddy/Caddyfile
agent.example.com {
    reverse_proxy 127.0.0.1:8443
}
```

Reload Caddy:
```bash
sudo systemctl reload caddy
```

---

### Option C: Nginx

If using Nginx, disable proxy buffering so that Server-Sent Events (SSE) stream to your client without delay:

```nginx
server {
    server_name agent.example.com;

    location / {
        proxy_pass http://127.0.0.1:8443;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # SSE Streaming Settings:
        proxy_http_version 1.1;
        proxy_set_header Connection '';
        proxy_buffering off;
        proxy_cache off;
        chunked_transfer_encoding off;
        proxy_read_timeout 600s;
    }

    listen 443 ssl;
    # ssl_certificate /path/to/cert.pem;
    # ssl_certificate_key /path/to/key.pem;
}
```

---

## 4. API Reference

All requests must include the authorization token:
- Header: `Authorization: Bearer <TOKEN>`
- Or Query parameter: `?token=<TOKEN>`

### 1. Health & Workspace Check

Verify server reachability and active model.

```bash
curl -X GET "https://agent.example.com/api/health" \
  -H "Authorization: Bearer <TOKEN>"
```

**Response (`200 OK`)**:
```json
{
  "status": "ok",
  "repo_root": "/home/user/Projects/omarchy-agent",
  "model": "qwen3.8-4b-q4"
}
```

---

### 2. View Tasks

Retrieve all persistent tasks stored in the SQLite database.

```bash
curl -X GET "https://agent.example.com/api/tasks" \
  -H "Authorization: Bearer <TOKEN>"
```

**Response (`200 OK`)**:
```json
{
  "tasks": [
    {
      "id": 1,
      "title": "Step 1: Implement terminal execution bridge",
      "description": "Create terminal.py with timeout and path checks",
      "status": "done",
      "created_at": "2026-11-03T12:00:00Z",
      "updated_at": "2026-11-03T12:15:00Z"
    }
  ]
}
```

---

### 3. Chat & Remote Prompt Execution

#### Standard JSON Response
Send a prompt and receive the completed response:

```bash
curl -X POST "https://agent.example.com/api/chat" \
  -H "Authorization: Bearer <TOKEN>" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Inspect the repository status and summarize changes."}'
```

**Response (`200 OK`)**:
```json
{
  "answer": "The working tree is clean on branch main with 330 tests passing."
}
```

#### Real-time SSE Streaming
For immediate token-by-token streaming, pass `"stream": true` or request header `Accept: text/event-stream`:

```bash
curl -N -X POST "https://agent.example.com/api/chat" \
  -H "Authorization: Bearer <TOKEN>" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Deconstruct a plan for the new test bridge", "stream": true}'
```

**Stream chunks received**:
```text
data: {"delta": "Here "}

data: {"delta": "is "}

data: {"delta": "the proposed "}

data: {"delta": "plan:\n"}

...

data: {"done": true, "answer": "Here is the proposed plan: ..."}

data: [DONE]
```

---

## 5. Client Integration Examples

### Python Client

```python
import json
import urllib.request

TOKEN = "my-secure-remote-token"
BASE_URL = "https://agent.example.com"


def prompt_agent(prompt: str) -> str:
    url = f"{BASE_URL}/api/chat"
    headers = {
        "Authorization": f"Bearer {TOKEN}",
        "Content-Type": "application/json",
    }
    payload = json.dumps({"prompt": prompt}).encode("utf-8")
    req = urllib.request.Request(url, data=payload, headers=headers)
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read().decode("utf-8"))
        return data["answer"]


if __name__ == "__main__":
    print(prompt_agent("What capabilities are currently allowed?"))
```

### JavaScript / Browser Client (Fetch SSE)

```javascript
async function streamAgent(prompt, token) {
  const response = await fetch("https://agent.example.com/api/chat", {
    method: "POST",
    headers: {
      "Authorization": `Bearer ${token}`,
      "Content-Type": "application/json"
    },
    body: JSON.stringify({ prompt, stream: true })
  });

  const reader = response.body.getReader();
  const decoder = new TextDecoder();

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    const text = decoder.decode(value);
    for (const line of text.split("\n")) {
      if (line.startsWith("data: ") && line !== "data: [DONE]") {
        const payload = JSON.parse(line.slice(6));
        if (payload.delta) {
          process.stdout.write(payload.delta);
        }
      }
    }
  }
}
```

---

## 6. Running as a Background Service (systemd)

To keep the remote server running persistently in the background:

1. Create a user systemd service file:
   `~/.config/systemd/user/omarchy-remote.service`

   ```ini
   [Unit]
   Description=Omarchy Agent Remote Access Server
   After=network.target

   [Service]
   Type=simple
   WorkingDirectory=/home/rithvin007/Projects/omarchy-agent
   ExecStart=/home/rithvin007/Projects/omarchy-agent/.venv/bin/omarchy-agent serve --port 8443 --token YOUR_TOKEN
   Restart=always
   RestartSec=5

   [Install]
   WantedBy=default.target
   ```

2. Enable and start the service:
   ```bash
   systemctl --user daemon-reload
   systemctl --user enable --now omarchy-remote.service
   ```

3. Check service status:
   ```bash
   systemctl --user status omarchy-remote.service
   ```

