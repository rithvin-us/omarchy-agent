"""Google OAuth (loopback + PKCE) and minimal Gmail/Calendar REST client.

Stdlib only. The user supplies an OAuth "Desktop app" client JSON at
``~/.config/omarchy-agent/google_client.json``; tokens are stored 0600 at
``~/.local/share/omarchy-agent/google_token.json``.
"""

from __future__ import annotations

import base64
import hashlib
import http.server
import json
import os
import secrets
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import webbrowser
from datetime import datetime
from pathlib import Path
from typing import Any

__all__ = ["GoogleAuthError", "GoogleClient"]

SCOPES = (
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/calendar.events",
)
CLIENT_PATH = Path("~/.config/omarchy-agent/google_client.json").expanduser()
TOKEN_PATH = Path("~/.local/share/omarchy-agent/google_token.json").expanduser()
AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN_URL = "https://oauth2.googleapis.com/token"  # noqa: S105
MAX_BODY_CHARS = 4000


class GoogleAuthError(RuntimeError):
    """Not logged in, bad credentials, or an API error."""


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


class GoogleClient:
    def __init__(self, client_path: Path = CLIENT_PATH, token_path: Path = TOKEN_PATH) -> None:
        self.client_path = client_path
        self.token_path = token_path

    # -- auth ------------------------------------------------------------
    def _client_config(self) -> dict[str, Any]:
        try:
            raw = json.loads(self.client_path.read_text())
        except (OSError, ValueError) as exc:
            raise GoogleAuthError(
                f"missing OAuth client file {self.client_path} (Google Cloud Console, "
                "Credentials, OAuth client ID, Desktop app, download JSON)"
            ) from exc
        cfg = raw.get("installed") or raw.get("web") or raw
        if "client_id" not in cfg:
            raise GoogleAuthError("client JSON has no client_id")
        return cfg

    def _save_token(self, token: dict[str, Any]) -> None:
        self.token_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.token_path.with_suffix(".tmp")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as handle:
            json.dump(token, handle)
        tmp.replace(self.token_path)

    def _post_token(self, data: dict[str, str]) -> dict[str, Any]:
        req = urllib.request.Request(  # noqa: S310
            TOKEN_URL, data=urllib.parse.urlencode(data).encode(), method="POST"
        )
        try:
            with urllib.request.urlopen(req, timeout=20) as resp:  # noqa: S310
                return json.load(resp)
        except urllib.error.HTTPError as exc:
            raise GoogleAuthError(f"token endpoint HTTP {exc.code}") from exc
        except (urllib.error.URLError, OSError, ValueError) as exc:
            raise GoogleAuthError(f"token endpoint failed: {exc}") from exc

    def login(self, *, open_browser: bool = True, timeout: float = 180.0) -> str:
        """Interactive loopback login. Returns the authorize URL used."""
        cfg = self._client_config()
        verifier = _b64(secrets.token_bytes(48))
        challenge = _b64(hashlib.sha256(verifier.encode()).digest())
        state = secrets.token_urlsafe(16)
        result: dict[str, str] = {}

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802
                query = dict(urllib.parse.parse_qsl(urllib.parse.urlparse(self.path).query))
                result.update(query)
                self.send_response(200)
                self.send_header("Content-Type", "text/plain")
                self.end_headers()
                self.wfile.write(b"omarchy-agent: login complete, you can close this tab.")

            def log_message(self, *_args: object) -> None:
                return

        server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
        redirect = f"http://127.0.0.1:{server.server_port}"
        url = (
            AUTH_URL
            + "?"
            + urllib.parse.urlencode(
                {
                    "client_id": cfg["client_id"],
                    "redirect_uri": redirect,
                    "response_type": "code",
                    "scope": " ".join(SCOPES),
                    "access_type": "offline",
                    "prompt": "consent",
                    "state": state,
                    "code_challenge": challenge,
                    "code_challenge_method": "S256",
                }
            )
        )
        if open_browser:
            webbrowser.open(url)
        server.timeout = timeout
        thread = threading.Thread(target=server.handle_request, daemon=True)
        thread.start()
        thread.join(timeout)
        server.server_close()
        if result.get("state") != state or "code" not in result:
            raise GoogleAuthError("login failed or timed out")
        body = {
            "client_id": cfg["client_id"],
            "code": result["code"],
            "code_verifier": verifier,
            "redirect_uri": redirect,
            "grant_type": "authorization_code",
        }
        if cfg.get("client_secret"):
            body["client_secret"] = cfg["client_secret"]
        token = self._post_token(body)
        token["expires_at"] = time.time() + int(token.get("expires_in", 3600))
        self._save_token(token)
        return url

    def _access_token(self) -> str:
        try:
            token = json.loads(self.token_path.read_text())
        except (OSError, ValueError) as exc:
            raise GoogleAuthError("not logged in; run: omarchy-agent google-login") from exc
        if token.get("expires_at", 0) > time.time() + 60:
            return str(token["access_token"])
        refresh = token.get("refresh_token")
        if not refresh:
            raise GoogleAuthError("no refresh token; run: omarchy-agent google-login")
        cfg = self._client_config()
        body = {
            "client_id": cfg["client_id"],
            "refresh_token": refresh,
            "grant_type": "refresh_token",
        }
        if cfg.get("client_secret"):
            body["client_secret"] = cfg["client_secret"]
        fresh = self._post_token(body)
        token.update(fresh)
        token["expires_at"] = time.time() + int(fresh.get("expires_in", 3600))
        self._save_token(token)
        return str(token["access_token"])

    def _request(
        self, method: str, url: str, params: dict[str, Any] | None = None, body: Any = None
    ) -> Any:
        if not url.startswith("https://"):
            raise GoogleAuthError("refusing non-https URL")
        if params:
            url += "?" + urllib.parse.urlencode(params)
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(  # noqa: S310
            url,
            data=data,
            method=method,
            headers={
                "Authorization": f"Bearer {self._access_token()}",
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=20) as resp:  # noqa: S310
                raw = resp.read()
        except urllib.error.HTTPError as exc:
            raise GoogleAuthError(f"Google API HTTP {exc.code}") from exc
        except (urllib.error.URLError, OSError) as exc:
            raise GoogleAuthError(f"Google API failed: {exc}") from exc
        return json.loads(raw) if raw else {}

    # -- gmail -----------------------------------------------------------
    def gmail_search(self, query: str, limit: int = 10) -> list[dict[str, str]]:
        base = "https://gmail.googleapis.com/gmail/v1/users/me/messages"
        listing = self._request("GET", base, {"q": query, "maxResults": min(limit, 20)})
        out = []
        for ref in listing.get("messages", []):
            msg = self._request(
                "GET",
                f"{base}/{urllib.parse.quote(ref['id'], safe='')}",
                {"format": "metadata", "metadataHeaders": ["Subject", "From", "Date"]},
            )
            headers = {h["name"]: h["value"] for h in msg.get("payload", {}).get("headers", [])}
            out.append(
                {
                    "id": ref["id"],
                    "from": headers.get("From", ""),
                    "subject": headers.get("Subject", ""),
                    "date": headers.get("Date", ""),
                    "snippet": msg.get("snippet", ""),
                }
            )
        return out

    def gmail_read(self, message_id: str) -> dict[str, str]:
        msg = self._request(
            "GET",
            "https://gmail.googleapis.com/gmail/v1/users/me/messages/"
            + urllib.parse.quote(message_id, safe=""),
            {"format": "full"},
        )
        headers = {h["name"]: h["value"] for h in msg.get("payload", {}).get("headers", [])}
        return {
            "id": message_id,
            "from": headers.get("From", ""),
            "subject": headers.get("Subject", ""),
            "date": headers.get("Date", ""),
            "body": _extract_text(msg.get("payload", {}))[:MAX_BODY_CHARS],
        }

    # -- calendar --------------------------------------------------------
    def calendar_list(self, time_min: str, time_max: str, limit: int = 25) -> list[dict[str, Any]]:
        data = self._request(
            "GET",
            "https://www.googleapis.com/calendar/v3/calendars/primary/events",
            {
                "timeMin": _rfc3339(time_min),
                "timeMax": _rfc3339(time_max),
                "singleEvents": "true",
                "orderBy": "startTime",
                "maxResults": min(limit, 100),
            },
        )
        return [
            {
                "id": e.get("id"),
                "summary": e.get("summary", ""),
                "start": e.get("start", {}).get("dateTime") or e.get("start", {}).get("date"),
                "end": e.get("end", {}).get("dateTime") or e.get("end", {}).get("date"),
            }
            for e in data.get("items", [])
        ]

    def calendar_create(
        self, summary: str, start: str, end: str, description: str = ""
    ) -> dict[str, Any]:
        e = self._request(
            "POST",
            "https://www.googleapis.com/calendar/v3/calendars/primary/events",
            body={
                "summary": summary,
                "description": description,
                "start": {"dateTime": _rfc3339(start)},
                "end": {"dateTime": _rfc3339(end)},
            },
        )
        return {"id": e.get("id"), "link": e.get("htmlLink")}

    def calendar_delete(self, event_id: str) -> dict[str, Any]:
        self._request(
            "DELETE",
            "https://www.googleapis.com/calendar/v3/calendars/primary/events/"
            + urllib.parse.quote(event_id, safe=""),
        )
        return {"deleted": event_id}


def _rfc3339(value: str) -> str:
    """Accept ISO-8601; naive times are taken as local time."""
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise GoogleAuthError(f"not an ISO-8601 time: {value!r}") from exc
    if parsed.tzinfo is None:
        parsed = parsed.astimezone()
    return parsed.isoformat()


def _extract_text(payload: dict[str, Any]) -> str:
    """Prefer text/plain; fall back to the first decodable part."""
    stack = [payload]
    fallback = ""
    while stack:
        part = stack.pop(0)
        data = part.get("body", {}).get("data")
        if data:
            text = base64.urlsafe_b64decode(data + "=" * (-len(data) % 4)).decode(
                "utf-8", "replace"
            )
            if part.get("mimeType") == "text/plain":
                return text
            fallback = fallback or text
        stack.extend(part.get("parts", []))
    return fallback
