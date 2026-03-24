"""
System Info Tool — monitor hardware, processes, and system state.
"""

import subprocess
import psutil
from typing import Callable

from tools.base import BaseTool


class SystemInfoTool(BaseTool):
    
    def system_stats(self) -> str:
        """Get CPU, RAM, GPU, and disk stats."""
        lines = ["System Stats:\n"]
        
        # CPU
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_count = psutil.cpu_count()
        cpu_freq = psutil.cpu_freq()
        lines.append(f"CPU: {cpu_percent}% usage ({cpu_count} cores)")
        if cpu_freq:
            lines.append(f"     {cpu_freq.current:.0f}MHz / {cpu_freq.max:.0f}MHz max")
        
        # RAM
        mem = psutil.virtual_memory()
        lines.append(f"RAM: {mem.percent}% used ({self._h(mem.used)} / {self._h(mem.total)})")
        lines.append(f"     Available: {self._h(mem.available)}")
        
        # Swap
        swap = psutil.swap_memory()
        if swap.total > 0:
            lines.append(f"Swap: {swap.percent}% ({self._h(swap.used)} / {self._h(swap.total)})")
        
        # Disk
        disk = psutil.disk_usage("/")
        lines.append(f"Disk /: {disk.percent}% ({self._h(disk.used)} / {self._h(disk.total)})")
        
        # GPU (nvidia-smi)
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                for line in result.stdout.strip().split("\n"):
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) == 5:
                        name, util, mem_used, mem_total, temp = parts
                        lines.append(f"GPU: {name} — {util}% util, {mem_used}/{mem_total} MB VRAM, {temp}°C")
        except FileNotFoundError:
            pass
        
        # Uptime
        try:
            result = subprocess.run(["uptime", "-p"], capture_output=True, text=True, timeout=5)
            lines.append(f"Uptime: {result.stdout.strip()}")
        except Exception:
            pass
        
        return "\n".join(lines)
    
    def list_processes(self, sort_by: str = "cpu", count: int = 15) -> str:
        """List top processes by CPU or memory usage."""
        procs = []
        for p in psutil.process_iter(["pid", "name", "cpu_percent", "memory_percent", "status"]):
            try:
                info = p.info
                procs.append(info)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        
        key = "cpu_percent" if sort_by == "cpu" else "memory_percent"
        procs.sort(key=lambda x: x.get(key, 0) or 0, reverse=True)
        
        lines = [f"Top {count} processes by {'CPU' if sort_by == 'cpu' else 'Memory'}:\n"]
        lines.append(f"  {'PID':>7}  {'CPU%':>6}  {'MEM%':>6}  {'STATUS':<12}  NAME")
        lines.append("  " + "─" * 55)
        
        for p in procs[:count]:
            lines.append(
                f"  {p['pid']:>7}  {(p.get('cpu_percent') or 0):>5.1f}%  "
                f"{(p.get('memory_percent') or 0):>5.1f}%  {p.get('status', '?'):<12}  {p.get('name', '?')}"
            )
        
        return "\n".join(lines)
    
    def network_info(self) -> str:
        """Get network interface and connection info."""
        lines = ["Network Info:\n"]
        
        # Interfaces
        addrs = psutil.net_if_addrs()
        stats = psutil.net_if_stats()
        for iface, addr_list in addrs.items():
            st = stats.get(iface)
            if st and st.isup:
                for addr in addr_list:
                    if addr.family.name == "AF_INET":
                        lines.append(f"  {iface}: {addr.address} (up, {st.speed}Mbps)")
        
        # IO counters
        io = psutil.net_io_counters()
        lines.append(f"\nTraffic: {self._h(io.bytes_sent)} sent, {self._h(io.bytes_recv)} received")
        
        # External IP
        try:
            result = subprocess.run(
                ["curl", "-s", "ifconfig.me"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                lines.append(f"External IP: {result.stdout.strip()}")
        except Exception:
            pass
        
        return "\n".join(lines)
    
    def _h(self, size: int) -> str:
        for unit in ["B", "KB", "MB", "GB", "TB"]:
            if abs(size) < 1024:
                return f"{size:.1f}{unit}"
            size /= 1024
        return f"{size:.1f}PB"
    
    def get_tool_definitions(self) -> list[dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "system_stats",
                    "description": "Get current CPU, RAM, GPU, disk usage statistics.",
                    "parameters": {"type": "object", "properties": {}}
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "list_processes",
                    "description": "List top running processes sorted by CPU or memory usage.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "sort_by": {"type": "string", "enum": ["cpu", "memory"]},
                            "count": {"type": "integer", "description": "Number of processes to show"}
                        }
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "network_info",
                    "description": "Get network interfaces, IP addresses, and traffic stats.",
                    "parameters": {"type": "object", "properties": {}}
                }
            }
        ]
    
    def get_handlers(self) -> dict[str, Callable]:
        return {
            "system_stats": self.system_stats,
            "list_processes": self.list_processes,
            "network_info": self.network_info,
        }
