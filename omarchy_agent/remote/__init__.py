"""Remote access server package for Omarchy Agent.

Provides a reverse-proxy friendly, token-authenticated HTTP/SSE server.
"""

from .server import RemoteServer, create_remote_server

__all__ = ["RemoteServer", "create_remote_server"]
