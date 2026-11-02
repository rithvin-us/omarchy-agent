"""Read-only system information bridge.

Everything here is observation only: no configuration is read for the purpose
of changing it, no privileged interface is touched, and nothing is installed.

Deliberate omissions, because they commonly carry secrets:

* process command lines (``/proc/<pid>/cmdline``) are never read -- only the
  kernel ``comm`` name;
* environment blocks (``/proc/<pid>/environ``) are never read;
* ``os.environ`` is never returned.

External commands are invoked with an argument array, a timeout and
``shell=False`` (the default). ``nvidia-smi`` is used only if it already
exists on ``PATH``.
"""

from __future__ import annotations

import contextlib
import os
import platform
import shutil
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..security.audit import audited
from ..security.permissions import Capability

__all__ = [
    "CpuInformation",
    "GpuDevice",
    "GpuInformation",
    "MemoryInformation",
    "OsInformation",
    "ProcessSummary",
    "StorageInformation",
    "StorageVolume",
    "get_cpu_information",
    "get_gpu_information",
    "get_memory_information",
    "get_os_information",
    "get_running_process_summary",
    "get_storage_information",
    "run_readonly_command",
]

#: Timeout applied to every external command this module runs.
COMMAND_TIMEOUT_SECONDS = 5.0

#: Filesystem types that are not real storage and are excluded from reports.
PSEUDO_FILESYSTEMS: frozenset[str] = frozenset(
    {
        "autofs",
        "bpf",
        "cgroup",
        "cgroup2",
        "configfs",
        "debugfs",
        "devpts",
        "devtmpfs",
        "efivarfs",
        "fuse.gvfsd-fuse",
        "fuse.portal",
        "fusectl",
        "hugetlbfs",
        "mqueue",
        "proc",
        "pstore",
        "ramfs",
        "securityfs",
        "sysfs",
        "tracefs",
        "binfmt_misc",
        "nsfs",
        "overlay",
        "squashfs",
        "tmpfs",
    }
)


class SystemInformationError(RuntimeError):
    """Raised when a system information source is unavailable."""


# --------------------------------------------------------------------------
# subprocess helper
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class CommandResult:
    executable: str
    available: bool
    returncode: int | None
    stdout: str
    stderr: str
    timed_out: bool
    error: str | None

    def ok(self) -> bool:
        return self.available and not self.timed_out and self.returncode == 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "executable": self.executable,
            "available": self.available,
            "returncode": self.returncode,
            "timed_out": self.timed_out,
            "error": self.error,
        }


def run_readonly_command(
    argv: list[str],
    *,
    timeout: float = COMMAND_TIMEOUT_SECONDS,
    max_output_chars: int = 64 * 1024,
) -> CommandResult:
    """Run a fixed, read-only command.

    ``argv`` must be a list whose first element is an executable name resolved
    through :func:`shutil.which`. ``shell=False`` is the default and is never
    overridden anywhere in this project. Nothing in ``argv`` may come from a
    language model or from untrusted text: callers in this module pass literal
    lists only.
    """
    if not argv:
        raise ValueError("argv must not be empty")
    if any(not isinstance(part, str) for part in argv):
        raise ValueError("every argv element must be a string")

    executable = shutil.which(argv[0])
    if executable is None:
        return CommandResult(
            executable=argv[0],
            available=False,
            returncode=None,
            stdout="",
            stderr="",
            timed_out=False,
            error=f"{argv[0]} not found on PATH",
        )

    try:
        completed = subprocess.run(  # noqa: S603 - fixed argv, shell=False
            [executable, *argv[1:]],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            env={"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "LC_ALL": "C"},
        )
    except subprocess.TimeoutExpired:
        return CommandResult(
            executable=executable,
            available=True,
            returncode=None,
            stdout="",
            stderr="",
            timed_out=True,
            error=f"timed out after {timeout}s",
        )
    except OSError as exc:
        return CommandResult(
            executable=executable,
            available=True,
            returncode=None,
            stdout="",
            stderr="",
            timed_out=False,
            error=f"{type(exc).__name__}: {exc}",
        )

    return CommandResult(
        executable=executable,
        available=True,
        returncode=completed.returncode,
        stdout=(completed.stdout or "")[:max_output_chars],
        stderr=(completed.stderr or "")[:max_output_chars],
        timed_out=False,
        error=None,
    )


# --------------------------------------------------------------------------
# /proc and /etc parsing helpers
# --------------------------------------------------------------------------


def _read_text_file(path: str | os.PathLike[str], limit: int = 256 * 1024) -> str | None:
    try:
        with open(path, encoding="utf-8", errors="replace") as handle:
            return handle.read(limit)
    except OSError:
        return None


def _parse_key_value_file(text: str) -> dict[str, str]:
    """Parse ``KEY=value`` lines, as used by ``/etc/os-release``."""
    parsed: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        parsed[key.strip()] = value.strip().strip('"').strip("'")
    return parsed


def _parse_meminfo(text: str) -> dict[str, int]:
    """Parse ``/proc/meminfo`` into a mapping of field name to bytes."""
    values: dict[str, int] = {}
    for line in text.splitlines():
        name, _, rest = line.partition(":")
        rest = rest.strip()
        if not rest:
            continue
        parts = rest.split()
        try:
            amount = int(parts[0])
        except (ValueError, IndexError):
            continue
        if len(parts) > 1 and parts[1].lower() == "kb":
            amount *= 1024
        values[name.strip()] = amount
    return values


# --------------------------------------------------------------------------
# result types
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class OsInformation:
    distribution: str
    distribution_version: str
    distribution_id: str
    kernel_name: str
    kernel_release: str
    architecture: str
    hostname: str
    python_version: str
    python_executable: str
    boot_time: str | None
    uptime_seconds: float | None
    desktop_session: str
    display_server: str
    is_omarchy: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "distribution": self.distribution,
            "distribution_version": self.distribution_version,
            "distribution_id": self.distribution_id,
            "kernel_name": self.kernel_name,
            "kernel_release": self.kernel_release,
            "architecture": self.architecture,
            "hostname": self.hostname,
            "python_version": self.python_version,
            "python_executable": self.python_executable,
            "boot_time": self.boot_time,
            "uptime_seconds": self.uptime_seconds,
            "desktop_session": self.desktop_session,
            "display_server": self.display_server,
            "is_omarchy": self.is_omarchy,
        }

    def audit_summary(self) -> str:
        return f"os={self.distribution} {self.distribution_version} kernel={self.kernel_release}"


@dataclass(frozen=True)
class CpuInformation:
    model: str
    vendor: str
    architecture: str
    physical_cores: int | None
    logical_cores: int | None
    usable_cores: int | None
    current_frequency_mhz: float | None
    max_frequency_mhz: float | None
    load_average: tuple[float, float, float] | None
    temperature_celsius: float | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "vendor": self.vendor,
            "architecture": self.architecture,
            "physical_cores": self.physical_cores,
            "logical_cores": self.logical_cores,
            "usable_cores": self.usable_cores,
            "current_frequency_mhz": self.current_frequency_mhz,
            "max_frequency_mhz": self.max_frequency_mhz,
            "load_average": list(self.load_average) if self.load_average else None,
            "temperature_celsius": self.temperature_celsius,
        }

    def audit_summary(self) -> str:
        return f"cpu={self.model} logical_cores={self.logical_cores}"


@dataclass(frozen=True)
class MemoryInformation:
    total_bytes: int
    available_bytes: int
    used_bytes: int
    free_bytes: int
    cached_bytes: int
    used_percent: float
    swap_total_bytes: int
    swap_used_bytes: int
    swap_used_percent: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "total_bytes": self.total_bytes,
            "available_bytes": self.available_bytes,
            "used_bytes": self.used_bytes,
            "free_bytes": self.free_bytes,
            "cached_bytes": self.cached_bytes,
            "used_percent": self.used_percent,
            "swap_total_bytes": self.swap_total_bytes,
            "swap_used_bytes": self.swap_used_bytes,
            "swap_used_percent": self.swap_used_percent,
        }

    def audit_summary(self) -> str:
        return f"memory total={self.total_bytes} used_percent={self.used_percent:.1f}"


@dataclass(frozen=True)
class StorageVolume:
    device: str
    mountpoint: str
    filesystem: str
    total_bytes: int
    used_bytes: int
    free_bytes: int
    used_percent: float
    read_only: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "device": self.device,
            "mountpoint": self.mountpoint,
            "filesystem": self.filesystem,
            "total_bytes": self.total_bytes,
            "used_bytes": self.used_bytes,
            "free_bytes": self.free_bytes,
            "used_percent": self.used_percent,
            "read_only": self.read_only,
        }


@dataclass(frozen=True)
class StorageInformation:
    volumes: list[StorageVolume]
    source: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "volumes": [volume.as_dict() for volume in self.volumes],
            "source": self.source,
        }

    def audit_summary(self) -> str:
        return f"storage: {len(self.volumes)} volumes (source={self.source})"


@dataclass(frozen=True)
class GpuDevice:
    index: int
    name: str
    driver_version: str
    memory_total_mib: int | None
    memory_used_mib: int | None
    utilization_percent: int | None
    temperature_celsius: int | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "name": self.name,
            "driver_version": self.driver_version,
            "memory_total_mib": self.memory_total_mib,
            "memory_used_mib": self.memory_used_mib,
            "utilization_percent": self.utilization_percent,
            "temperature_celsius": self.temperature_celsius,
        }


@dataclass(frozen=True)
class GpuInformation:
    available: bool
    source: str
    devices: list[GpuDevice] = field(default_factory=list)
    detail: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "source": self.source,
            "devices": [device.as_dict() for device in self.devices],
            "detail": self.detail,
        }

    def audit_summary(self) -> str:
        return f"gpu available={self.available} devices={len(self.devices)} source={self.source}"


@dataclass(frozen=True)
class ProcessSummary:
    total: int
    by_state: dict[str, int]
    top_by_memory: list[dict[str, Any]]
    thread_total: int
    source: str
    note: str = (
        "Process command lines and environment blocks are deliberately not "
        "collected because they frequently contain secrets."
    )

    def as_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "by_state": dict(self.by_state),
            "top_by_memory": list(self.top_by_memory),
            "thread_total": self.thread_total,
            "source": self.source,
            "note": self.note,
        }

    def audit_summary(self) -> str:
        return f"processes total={self.total} threads={self.thread_total}"


# --------------------------------------------------------------------------
# tools
# --------------------------------------------------------------------------


@audited("system.get_os_information", Capability.SYSTEM_INFO)
def get_os_information() -> OsInformation:
    """Return distribution, kernel, architecture and session information."""
    release = _parse_key_value_file(_read_text_file("/etc/os-release") or "")
    uname = platform.uname()

    boot_time: str | None = None
    uptime_seconds: float | None = None
    uptime_raw = _read_text_file("/proc/uptime")
    if uptime_raw:
        try:
            uptime_seconds = float(uptime_raw.split()[0])
            boot_time = datetime.fromtimestamp(time.time() - uptime_seconds, UTC).isoformat(
                timespec="seconds"
            )
        except (ValueError, IndexError):
            pass

    distribution_id = release.get("ID", "")
    return OsInformation(
        distribution=release.get("PRETTY_NAME") or release.get("NAME") or uname.system,
        distribution_version=release.get("VERSION_ID") or release.get("BUILD_ID") or "",
        distribution_id=distribution_id,
        kernel_name=uname.system,
        kernel_release=uname.release,
        architecture=platform.machine(),
        hostname=socket.gethostname(),
        python_version=platform.python_version(),
        python_executable=sys.executable,
        boot_time=boot_time,
        uptime_seconds=uptime_seconds,
        # These two identify the session type; neither is a credential.
        desktop_session=os.environ.get("XDG_CURRENT_DESKTOP", "unknown"),
        display_server=os.environ.get("XDG_SESSION_TYPE", "unknown"),
        is_omarchy=distribution_id == "omarchy",
    )


@audited("system.get_cpu_information", Capability.SYSTEM_INFO)
def get_cpu_information() -> CpuInformation:
    """Return CPU model, core counts, frequency and load from ``/proc``."""
    cpuinfo = _read_text_file("/proc/cpuinfo") or ""

    model = ""
    vendor = ""
    physical_ids: set[str] = set()
    core_ids: set[tuple[str, str]] = set()
    logical = 0
    frequencies: list[float] = []

    current_physical_id = ""
    for line in cpuinfo.splitlines():
        key, _, value = line.partition(":")
        key = key.strip().lower()
        value = value.strip()
        if key == "processor":
            logical += 1
        elif key == "model name" and not model:
            model = value
        elif key == "vendor_id" and not vendor:
            vendor = value
        elif key == "physical id":
            current_physical_id = value
            physical_ids.add(value)
        elif key == "core id":
            core_ids.add((current_physical_id, value))
        elif key == "cpu mhz":
            with contextlib.suppress(ValueError):
                frequencies.append(float(value))

    max_frequency: float | None = None
    raw_max = _read_text_file("/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq")
    if raw_max:
        with contextlib.suppress(ValueError):
            max_frequency = int(raw_max.strip()) / 1000.0

    load: tuple[float, float, float] | None
    try:
        one, five, fifteen = os.getloadavg()
        load = (round(one, 2), round(five, 2), round(fifteen, 2))
    except OSError:
        load = None

    temperature: float | None = None
    for zone in sorted(Path("/sys/class/thermal").glob("thermal_zone*")):
        kind = (_read_text_file(zone / "type") or "").strip()
        if kind in {"k10temp", "x86_pkg_temp", "acpitz", "cpu-thermal"}:
            raw_temp = _read_text_file(zone / "temp")
            if raw_temp:
                try:
                    temperature = int(raw_temp.strip()) / 1000.0
                    break
                except ValueError:
                    continue

    usable: int | None
    try:
        usable = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        usable = os.cpu_count()

    return CpuInformation(
        model=model or platform.processor() or "unknown",
        vendor=vendor or "unknown",
        architecture=platform.machine(),
        physical_cores=len(core_ids) or None,
        logical_cores=logical or os.cpu_count(),
        usable_cores=usable,
        current_frequency_mhz=round(sum(frequencies) / len(frequencies), 1)
        if frequencies
        else None,
        max_frequency_mhz=round(max_frequency, 1) if max_frequency else None,
        load_average=load,
        temperature_celsius=round(temperature, 1) if temperature is not None else None,
    )


@audited("system.get_memory_information", Capability.SYSTEM_INFO)
def get_memory_information() -> MemoryInformation:
    """Return RAM and swap usage from ``/proc/meminfo``."""
    text = _read_text_file("/proc/meminfo")
    if not text:
        raise SystemInformationError("/proc/meminfo is not readable on this system")

    values = _parse_meminfo(text)
    total = values.get("MemTotal", 0)
    available = values.get("MemAvailable", values.get("MemFree", 0))
    free = values.get("MemFree", 0)
    cached = values.get("Cached", 0) + values.get("SReclaimable", 0)
    used = max(0, total - available)
    swap_total = values.get("SwapTotal", 0)
    swap_free = values.get("SwapFree", 0)
    swap_used = max(0, swap_total - swap_free)

    return MemoryInformation(
        total_bytes=total,
        available_bytes=available,
        used_bytes=used,
        free_bytes=free,
        cached_bytes=cached,
        used_percent=round(used / total * 100, 1) if total else 0.0,
        swap_total_bytes=swap_total,
        swap_used_bytes=swap_used,
        swap_used_percent=round(swap_used / swap_total * 100, 1) if swap_total else 0.0,
    )


@audited("system.get_storage_information", Capability.SYSTEM_INFO)
def get_storage_information(
    *, include_pseudo: bool = False, max_volumes: int = 32
) -> StorageInformation:
    """Return per-volume disk usage, read from ``/proc/mounts`` + ``statvfs``."""
    mounts_text = _read_text_file("/proc/mounts")
    volumes: list[StorageVolume] = []
    seen_devices: set[tuple[str, str]] = set()

    if mounts_text is None:
        usage = shutil.disk_usage("/")
        volumes.append(
            StorageVolume(
                device="unknown",
                mountpoint="/",
                filesystem="unknown",
                total_bytes=usage.total,
                used_bytes=usage.used,
                free_bytes=usage.free,
                used_percent=round(usage.used / usage.total * 100, 1) if usage.total else 0.0,
                read_only=False,
            )
        )
        return StorageInformation(volumes=volumes, source="shutil.disk_usage")

    for line in mounts_text.splitlines():
        parts = line.split()
        if len(parts) < 4:
            continue
        device, mountpoint, filesystem, options = parts[0], parts[1], parts[2], parts[3]
        mountpoint = mountpoint.replace("\\040", " ")
        if not include_pseudo and filesystem in PSEUDO_FILESYSTEMS:
            continue
        key = (device, mountpoint)
        if key in seen_devices:
            continue
        seen_devices.add(key)

        try:
            usage = shutil.disk_usage(mountpoint)
        except OSError:
            continue
        if usage.total == 0:
            continue

        volumes.append(
            StorageVolume(
                device=device,
                mountpoint=mountpoint,
                filesystem=filesystem,
                total_bytes=usage.total,
                used_bytes=usage.used,
                free_bytes=usage.free,
                used_percent=round(usage.used / usage.total * 100, 1),
                read_only="ro" in options.split(","),
            )
        )
        if len(volumes) >= max_volumes:
            break

    volumes.sort(key=lambda volume: volume.total_bytes, reverse=True)
    return StorageInformation(volumes=volumes, source="/proc/mounts")


#: Fixed, read-only nvidia-smi query. No part of it is caller-supplied.
NVIDIA_SMI_ARGV: list[str] = [
    "nvidia-smi",
    "--query-gpu=index,name,driver_version,memory.total,memory.used,utilization.gpu,temperature.gpu",
    "--format=csv,noheader,nounits",
]


@audited("system.get_gpu_information", Capability.SYSTEM_INFO)
def get_gpu_information(*, timeout: float = COMMAND_TIMEOUT_SECONDS) -> GpuInformation:
    """Return GPU details via ``nvidia-smi``, if that binary already exists.

    No GPU tooling is ever installed. When ``nvidia-smi`` is absent the result
    reports ``available=False`` instead of raising.
    """
    result = run_readonly_command(NVIDIA_SMI_ARGV, timeout=timeout)

    if not result.available:
        return GpuInformation(
            available=False,
            source="nvidia-smi",
            detail="nvidia-smi is not installed; no GPU tooling will be installed by this agent",
        )
    if result.timed_out:
        return GpuInformation(
            available=False, source="nvidia-smi", detail=f"nvidia-smi timed out after {timeout}s"
        )
    if not result.ok():
        detail = (result.stderr or result.error or "nvidia-smi failed").strip().splitlines()
        return GpuInformation(
            available=False,
            source="nvidia-smi",
            detail=detail[0][:300] if detail else "nvidia-smi failed",
        )

    def to_int(raw: str) -> int | None:
        raw = raw.strip()
        if not raw or raw.lower() in {"n/a", "[n/a]", "not supported"}:
            return None
        try:
            return int(float(raw))
        except ValueError:
            return None

    devices: list[GpuDevice] = []
    for line in result.stdout.strip().splitlines():
        fields = [part.strip() for part in line.split(",")]
        if len(fields) < 7:
            continue
        index = to_int(fields[0])
        devices.append(
            GpuDevice(
                index=index if index is not None else len(devices),
                name=fields[1],
                driver_version=fields[2],
                memory_total_mib=to_int(fields[3]),
                memory_used_mib=to_int(fields[4]),
                utilization_percent=to_int(fields[5]),
                temperature_celsius=to_int(fields[6]),
            )
        )

    return GpuInformation(
        available=bool(devices),
        source="nvidia-smi",
        devices=devices,
        detail=None if devices else "nvidia-smi returned no GPU rows",
    )


@audited("system.get_running_process_summary", Capability.SYSTEM_INFO)
def get_running_process_summary(*, top: int = 10) -> ProcessSummary:
    """Summarize running processes from ``/proc``.

    Only the kernel-reported ``comm`` name, state, thread count and resident
    set size are collected. Command lines and environment blocks are never
    opened, so no argument-borne secret can leak through this tool.
    """
    proc_root = Path("/proc")
    if not proc_root.is_dir():
        raise SystemInformationError("/proc is not available on this system")

    page_size = os.sysconf("SC_PAGE_SIZE") if hasattr(os, "sysconf") else 4096
    by_state: dict[str, int] = {}
    processes: list[dict[str, Any]] = []
    thread_total = 0

    for entry in proc_root.iterdir():
        if not entry.name.isdigit():
            continue
        stat_text = _read_text_file(entry / "stat", limit=4096)
        if not stat_text:
            continue
        # Format: pid (comm) state ... where comm may itself contain spaces.
        open_paren = stat_text.find("(")
        close_paren = stat_text.rfind(")")
        if open_paren == -1 or close_paren == -1 or close_paren < open_paren:
            continue
        name = stat_text[open_paren + 1 : close_paren]
        tail = stat_text[close_paren + 2 :].split()
        if not tail:
            continue
        state = tail[0]
        by_state[state] = by_state.get(state, 0) + 1
        try:
            threads = int(tail[17])
        except (IndexError, ValueError):
            threads = 1
        thread_total += threads

        rss_bytes = 0
        statm_text = _read_text_file(entry / "statm", limit=512)
        if statm_text:
            fields = statm_text.split()
            if len(fields) >= 2:
                try:
                    rss_bytes = int(fields[1]) * page_size
                except ValueError:
                    rss_bytes = 0

        processes.append(
            {
                "pid": int(entry.name),
                "name": name,
                "state": state,
                "threads": threads,
                "resident_bytes": rss_bytes,
            }
        )

    processes.sort(key=lambda item: item["resident_bytes"], reverse=True)
    return ProcessSummary(
        total=len(processes),
        by_state=dict(sorted(by_state.items())),
        top_by_memory=processes[: max(0, int(top))],
        thread_total=thread_total,
        source="/proc",
    )
