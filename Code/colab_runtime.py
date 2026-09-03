"""Colab session runtime tracking, tailored to paid (Pro/Pro+) GPU tiers.

Paid Colab runtimes can legitimately run for many hours. Without a periodic
backup, a disconnect near the end of a long paid session loses everything
computed since the last manual save. ColabRuntimeMonitor ties elapsed
wall-clock runtime to a Drive backup so long runs are protected, and reports
a runtime summary that can be recorded alongside the training metrics.
"""
import os
import shutil
import subprocess
import time
from datetime import timedelta
from typing import Dict, Iterable, Optional

# GPU names Colab hands out on Pro/Pro+ tiers; the free tier is almost
# always a T4. This is a heuristic, not an authoritative source.
PAID_GPU_NAMES = ('A100', 'V100', 'L4', 'A10G', 'P100')

# Colab's documented session ceilings, used only to size the printed budget
# estimate - nothing here enforces or extends them.
FREE_TIER_MAX_HOURS = 12
PRO_TIER_MAX_HOURS = 24


class ColabRuntimeMonitor:
    def __init__(self, backup_dir: Optional[str] = None, backup_interval_seconds: float = 1800):
        self.session_start = time.time()
        self.backup_dir = backup_dir
        self.backup_interval_seconds = backup_interval_seconds
        self._last_backup = self.session_start
        self.gpu_name, self.is_paid_tier = self._detect_runtime_tier()

    @staticmethod
    def _detect_runtime_tier():
        try:
            output = subprocess.check_output(
                ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
                stderr=subprocess.DEVNULL, text=True, timeout=5,
            )
            gpu_name = output.strip().splitlines()[0] if output.strip() else None
        except Exception:
            gpu_name = None
        is_paid_tier = bool(gpu_name) and any(tag in gpu_name for tag in PAID_GPU_NAMES)
        return gpu_name, is_paid_tier

    def elapsed_seconds(self) -> float:
        return time.time() - self.session_start

    def elapsed_str(self) -> str:
        return str(timedelta(seconds=int(self.elapsed_seconds())))

    def print_runtime_info(self):
        tier = "PAID (Pro/Pro+)" if self.is_paid_tier else "FREE / unknown"
        budget_hours = PRO_TIER_MAX_HOURS if self.is_paid_tier else FREE_TIER_MAX_HOURS
        print(f"{'='*60}")
        print(f"Colab runtime tier: {tier}")
        print(f"GPU: {self.gpu_name or 'none detected'}")
        print(f"Approx session budget: up to {budget_hours}h")
        if self.backup_dir:
            print(f"Auto-backup every {self.backup_interval_seconds / 60:.0f} min -> {self.backup_dir}")
        else:
            print("Auto-backup disabled (Google Drive not mounted)")
        print(f"{'='*60}")

    def checkpoint_due(self) -> bool:
        return bool(self.backup_dir) and (time.time() - self._last_backup) >= self.backup_interval_seconds

    def backup_results(self, source_dirs: Iterable[str]):
        """Copy source_dirs into self.backup_dir, if one was configured.

        Copies each directory into a staging path first and swaps it into
        place afterwards, rather than copying straight into the previous
        backup with dirs_exist_ok=True. A disconnect mid-copy then leaves
        the last known-good backup untouched instead of half-overwritten.
        """
        if not self.backup_dir:
            return
        os.makedirs(self.backup_dir, exist_ok=True)
        for src in source_dirs:
            if not os.path.isdir(src):
                continue
            name = os.path.basename(src)
            dest = os.path.join(self.backup_dir, name)
            staging = os.path.join(self.backup_dir, f'.{name}.staging')
            if os.path.exists(staging):
                shutil.rmtree(staging)
            shutil.copytree(src, staging)
            if os.path.exists(dest):
                shutil.rmtree(dest)
            os.rename(staging, dest)
        self._last_backup = time.time()
        print(f"[runtime {self.elapsed_str()}] backed up results to {self.backup_dir}")

    def summary(self) -> Dict:
        return {
            'session-runtime-seconds': round(self.elapsed_seconds(), 2),
            'session-runtime': self.elapsed_str(),
            'gpu': self.gpu_name,
            'paid-tier': self.is_paid_tier,
        }
