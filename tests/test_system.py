"""Tests for the read-only system information bridge.

External commands are mocked by default. The one test that touches the real
``nvidia-smi`` skips itself when that binary is absent.
"""

from __future__ import annotations

import shutil
import subprocess
import unittest
from unittest import mock

from omarchy_agent.bridge import system as system_bridge
from omarchy_agent.bridge.system import (
    CommandResult,
    get_cpu_information,
    get_gpu_information,
    get_memory_information,
    get_os_information,
    get_running_process_summary,
    get_storage_information,
    run_readonly_command,
)
from omarchy_agent.security.audit import AuditLogger, get_audit_logger, set_audit_logger

NVIDIA_SMI_SAMPLE = "0, NVIDIA GeForce RTX 4050 Laptop GPU, 580.95.05, 6141, 1024, 13, 45\n"


def fake_command_result(stdout: str = "", returncode: int = 0) -> CommandResult:
    return CommandResult(
        executable="/usr/bin/fake",
        available=True,
        returncode=returncode,
        stdout=stdout,
        stderr="",
        timed_out=False,
        error=None,
    )


class OsInformationTests(unittest.TestCase):
    def test_reports_distribution_and_kernel(self) -> None:
        info = get_os_information()
        self.assertTrue(info.kernel_name)
        self.assertTrue(info.kernel_release)
        self.assertTrue(info.architecture)
        self.assertTrue(info.hostname)
        self.assertTrue(info.python_version)

    def test_uptime_is_positive_when_available(self) -> None:
        info = get_os_information()
        if info.uptime_seconds is not None:
            self.assertGreater(info.uptime_seconds, 0)
            self.assertIsNotNone(info.boot_time)

    def test_dictionary_contains_no_environment_dump(self) -> None:
        payload = get_os_information().as_dict()
        self.assertNotIn("environment", payload)
        self.assertNotIn("env", payload)
        # Only the two session-type variables are surfaced, by name.
        self.assertEqual(
            {key for key in payload if "session" in key or "server" in key},
            {"desktop_session", "display_server"},
        )


class CpuInformationTests(unittest.TestCase):
    def test_reports_core_counts(self) -> None:
        info = get_cpu_information()
        self.assertIsNotNone(info.logical_cores)
        self.assertGreaterEqual(info.logical_cores or 0, 1)
        self.assertTrue(info.model)
        self.assertTrue(info.architecture)

    def test_load_average_has_three_values(self) -> None:
        info = get_cpu_information()
        if info.load_average is not None:
            self.assertEqual(len(info.load_average), 3)

    def test_handles_missing_proc_cpuinfo(self) -> None:
        with mock.patch.object(system_bridge, "_read_text_file", return_value=None):
            info = get_cpu_information()
        self.assertIsNotNone(info.logical_cores)
        self.assertTrue(info.model)

    def test_parses_a_synthetic_cpuinfo(self) -> None:
        sample = (
            "processor\t: 0\n"
            "vendor_id\t: AuthenticAMD\n"
            "model name\t: AMD Ryzen 7 7735HS with Radeon Graphics\n"
            "cpu MHz\t\t: 3194.000\n"
            "physical id\t: 0\n"
            "core id\t\t: 0\n"
            "\n"
            "processor\t: 1\n"
            "vendor_id\t: AuthenticAMD\n"
            "model name\t: AMD Ryzen 7 7735HS with Radeon Graphics\n"
            "cpu MHz\t\t: 3194.000\n"
            "physical id\t: 0\n"
            "core id\t\t: 1\n"
        )

        def fake_read(path, limit=256 * 1024):  # noqa: ANN001, ARG001
            return sample if str(path) == "/proc/cpuinfo" else None

        with mock.patch.object(system_bridge, "_read_text_file", side_effect=fake_read):
            info = get_cpu_information()
        self.assertEqual(info.logical_cores, 2)
        self.assertEqual(info.physical_cores, 2)
        self.assertEqual(info.vendor, "AuthenticAMD")
        self.assertIn("Ryzen", info.model)
        self.assertAlmostEqual(info.current_frequency_mhz or 0.0, 3194.0, places=1)


class MemoryInformationTests(unittest.TestCase):
    def test_reports_total_and_usage(self) -> None:
        info = get_memory_information()
        self.assertGreater(info.total_bytes, 0)
        self.assertGreaterEqual(info.used_bytes, 0)
        self.assertLessEqual(info.used_bytes, info.total_bytes)
        self.assertGreaterEqual(info.used_percent, 0.0)
        self.assertLessEqual(info.used_percent, 100.0)

    def test_parses_a_synthetic_meminfo(self) -> None:
        sample = (
            "MemTotal:       14000000 kB\n"
            "MemFree:         2000000 kB\n"
            "MemAvailable:    6000000 kB\n"
            "Cached:          4000000 kB\n"
            "SReclaimable:     500000 kB\n"
            "SwapTotal:       4000000 kB\n"
            "SwapFree:        3000000 kB\n"
        )
        with mock.patch.object(system_bridge, "_read_text_file", return_value=sample):
            info = get_memory_information()
        self.assertEqual(info.total_bytes, 14000000 * 1024)
        self.assertEqual(info.available_bytes, 6000000 * 1024)
        self.assertEqual(info.used_bytes, 8000000 * 1024)
        self.assertEqual(info.swap_used_bytes, 1000000 * 1024)
        self.assertAlmostEqual(info.used_percent, 57.1, places=1)

    def test_raises_when_meminfo_is_unavailable(self) -> None:
        with (
            mock.patch.object(system_bridge, "_read_text_file", return_value=None),
            self.assertRaises(system_bridge.SystemInformationError),
        ):
            get_memory_information()


class StorageInformationTests(unittest.TestCase):
    def test_reports_at_least_one_volume(self) -> None:
        info = get_storage_information()
        self.assertEqual(info.source, "/proc/mounts")
        self.assertTrue(info.volumes)
        for volume in info.volumes:
            self.assertGreater(volume.total_bytes, 0)
            self.assertGreaterEqual(volume.used_percent, 0.0)

    def test_excludes_pseudo_filesystems_by_default(self) -> None:
        info = get_storage_information()
        for volume in info.volumes:
            self.assertNotIn(volume.filesystem, {"proc", "sysfs", "devtmpfs", "tmpfs"})

    def test_falls_back_when_proc_mounts_is_unavailable(self) -> None:
        with mock.patch.object(system_bridge, "_read_text_file", return_value=None):
            info = get_storage_information()
        self.assertEqual(info.source, "shutil.disk_usage")
        self.assertEqual(len(info.volumes), 1)


class GpuInformationTests(unittest.TestCase):
    def test_parses_mocked_nvidia_smi_output(self) -> None:
        with mock.patch.object(
            system_bridge,
            "run_readonly_command",
            return_value=fake_command_result(NVIDIA_SMI_SAMPLE),
        ):
            info = get_gpu_information()
        self.assertTrue(info.available)
        self.assertEqual(len(info.devices), 1)
        device = info.devices[0]
        self.assertEqual(device.index, 0)
        self.assertIn("RTX 4050", device.name)
        self.assertEqual(device.driver_version, "580.95.05")
        self.assertEqual(device.memory_total_mib, 6141)
        self.assertEqual(device.memory_used_mib, 1024)
        self.assertEqual(device.utilization_percent, 13)
        self.assertEqual(device.temperature_celsius, 45)

    def test_handles_not_available_fields(self) -> None:
        sample = "0, Some GPU, 1.2.3, [N/A], N/A, N/A, N/A\n"
        with mock.patch.object(
            system_bridge, "run_readonly_command", return_value=fake_command_result(sample)
        ):
            info = get_gpu_information()
        device = info.devices[0]
        self.assertIsNone(device.memory_total_mib)
        self.assertIsNone(device.utilization_percent)

    def test_reports_unavailable_when_nvidia_smi_is_missing(self) -> None:
        missing = CommandResult(
            executable="nvidia-smi",
            available=False,
            returncode=None,
            stdout="",
            stderr="",
            timed_out=False,
            error="nvidia-smi not found on PATH",
        )
        with mock.patch.object(system_bridge, "run_readonly_command", return_value=missing):
            info = get_gpu_information()
        self.assertFalse(info.available)
        self.assertEqual(info.devices, [])
        self.assertIn("not installed", info.detail or "")

    def test_reports_timeout_without_raising(self) -> None:
        timed_out = CommandResult(
            executable="/usr/bin/nvidia-smi",
            available=True,
            returncode=None,
            stdout="",
            stderr="",
            timed_out=True,
            error="timed out after 5.0s",
        )
        with mock.patch.object(system_bridge, "run_readonly_command", return_value=timed_out):
            info = get_gpu_information()
        self.assertFalse(info.available)
        self.assertIn("timed out", info.detail or "")

    def test_reports_failure_without_raising(self) -> None:
        failed = CommandResult(
            executable="/usr/bin/nvidia-smi",
            available=True,
            returncode=9,
            stdout="",
            stderr="NVML: Driver/library version mismatch\n",
            timed_out=False,
            error=None,
        )
        with mock.patch.object(system_bridge, "run_readonly_command", return_value=failed):
            info = get_gpu_information()
        self.assertFalse(info.available)
        self.assertIn("NVML", info.detail or "")

    @unittest.skipIf(shutil.which("nvidia-smi") is None, "nvidia-smi is not installed")
    def test_real_nvidia_smi_is_parsed(self) -> None:
        info = get_gpu_information()
        self.assertEqual(info.source, "nvidia-smi")
        if info.available:
            self.assertTrue(info.devices)
            self.assertTrue(info.devices[0].name)
            self.assertTrue(info.devices[0].driver_version)
        else:
            self.assertIsNotNone(info.detail)


class ProcessSummaryTests(unittest.TestCase):
    def test_reports_processes(self) -> None:
        summary = get_running_process_summary(top=5)
        self.assertGreater(summary.total, 0)
        self.assertGreaterEqual(summary.thread_total, summary.total)
        self.assertLessEqual(len(summary.top_by_memory), 5)
        self.assertTrue(summary.by_state)

    def test_entries_have_no_command_line_field(self) -> None:
        summary = get_running_process_summary(top=3)
        for item in summary.top_by_memory:
            self.assertEqual(set(item), {"pid", "name", "state", "threads", "resident_bytes"})
            self.assertNotIn("cmdline", item)
            self.assertNotIn("environ", item)

    def test_top_is_sorted_by_resident_memory(self) -> None:
        summary = get_running_process_summary(top=10)
        sizes = [item["resident_bytes"] for item in summary.top_by_memory]
        self.assertEqual(sizes, sorted(sizes, reverse=True))

    def test_note_documents_the_omission(self) -> None:
        self.assertIn("command lines", get_running_process_summary(top=1).note.lower())


class RunReadonlyCommandTests(unittest.TestCase):
    def test_reports_a_missing_executable(self) -> None:
        result = run_readonly_command(["definitely-not-a-real-binary-xyz"])
        self.assertFalse(result.available)
        self.assertFalse(result.ok())
        self.assertIn("not found", result.error or "")

    def test_runs_a_real_harmless_command(self) -> None:
        result = run_readonly_command(["true"])
        self.assertTrue(result.available)
        self.assertEqual(result.returncode, 0)
        self.assertTrue(result.ok())

    def test_captures_a_nonzero_exit(self) -> None:
        result = run_readonly_command(["false"])
        self.assertTrue(result.available)
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(result.ok())

    def test_timeout_is_reported_not_raised(self) -> None:
        with mock.patch(
            "subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd=["sleep"], timeout=0.01),
        ):
            result = run_readonly_command(["true"], timeout=0.01)
        self.assertTrue(result.timed_out)
        self.assertFalse(result.ok())
        self.assertIn("timed out", result.error or "")

    def test_oserror_is_reported_not_raised(self) -> None:
        with mock.patch("subprocess.run", side_effect=OSError("exec format error")):
            result = run_readonly_command(["true"])
        self.assertFalse(result.ok())
        self.assertIn("OSError", result.error or "")

    def test_rejects_empty_argv(self) -> None:
        with self.assertRaises(ValueError):
            run_readonly_command([])

    def test_rejects_non_string_argv(self) -> None:
        with self.assertRaises(ValueError):
            run_readonly_command(["true", 7])  # type: ignore[list-item]

    def test_never_uses_a_shell(self) -> None:
        with mock.patch("subprocess.run", wraps=subprocess.run) as spy:
            run_readonly_command(["true"])
        _args, kwargs = spy.call_args
        self.assertNotIn("shell", kwargs)

    def test_output_is_length_capped(self) -> None:
        long_output = "y" * 5000
        completed = subprocess.CompletedProcess(
            args=["true"], returncode=0, stdout=long_output, stderr=""
        )
        with mock.patch("subprocess.run", return_value=completed):
            result = run_readonly_command(["true"], max_output_chars=100)
        self.assertEqual(len(result.stdout), 100)


class SystemAuditTests(unittest.TestCase):
    def setUp(self) -> None:
        self.previous = get_audit_logger()
        self.logger = AuditLogger()
        set_audit_logger(self.logger)

    def tearDown(self) -> None:
        set_audit_logger(self.previous)

    def test_every_system_call_is_audited(self) -> None:
        get_os_information()
        get_cpu_information()
        get_memory_information()
        get_storage_information()
        get_gpu_information()
        get_running_process_summary(top=1)
        names = [entry.tool_name for entry in self.logger.entries()]
        self.assertEqual(
            names,
            [
                "system.get_os_information",
                "system.get_cpu_information",
                "system.get_memory_information",
                "system.get_storage_information",
                "system.get_gpu_information",
                "system.get_running_process_summary",
            ],
        )
        for entry in self.logger.entries():
            self.assertEqual(entry.status, "success")
            self.assertTrue(entry.result_summary)


if __name__ == "__main__":
    unittest.main()
