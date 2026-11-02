"""Security primitives: path policy, capability policy, audit logging."""

from .audit import AuditEntry, AuditLogger, audited, get_audit_logger, set_audit_logger
from .paths import MAX_FILE_BYTES, PathPolicy, PathSecurityError, default_policy
from .permissions import (
    Capability,
    CapabilityNotImplemented,
    CapabilityState,
    PermissionDenied,
    PermissionPolicy,
    default_permission_policy,
)

__all__ = [
    "MAX_FILE_BYTES",
    "AuditEntry",
    "AuditLogger",
    "Capability",
    "CapabilityNotImplemented",
    "CapabilityState",
    "PathPolicy",
    "PathSecurityError",
    "PermissionDenied",
    "PermissionPolicy",
    "audited",
    "default_permission_policy",
    "default_policy",
    "get_audit_logger",
    "set_audit_logger",
]
