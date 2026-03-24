"""
Desktop Control Tool — control the Linux desktop environment.

Capabilities:
- Launch applications
- Manage windows (focus, move, resize, close)
- Clipboard read/write
- Take screenshots
- Run shell commands
- Type text / send keystrokes
"""

import subprocess
import shutil
import os
from pathlib import Path
from typing import Callable
from datetime import datetime

from tools.base import BaseTool


class DesktopControlTool(BaseTool):
    
    def launch_app(self, app_name: str) -> str:
        """Launch an application by name."""
        try:
            # Try direct command first
            subprocess.Popen(
                app_name, shell=True,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                start_new_session=True
            )
            return f"Launched: {app_name}"
        except Exception:
            # Try xdg-open for file types / URLs
            try:
                subprocess.Popen(
                    ["xdg-open", app_name],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    start_new_session=True
                )
                return f"Opened with xdg-open: {app_name}"
            except Exception as e:
                return f"Failed to launch {app_name}: {e}"
    
    def list_windows(self) -> str:
        """List all open windows."""
        try:
            result = subprocess.run(
                ["wmctrl", "-l", "-p"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode != 0:
                return "wmctrl not available. Install with: sudo apt install wmctrl"
            
            lines = ["Open windows:\n"]
            for line in result.stdout.strip().split("\n"):
                if line.strip():
                    parts = line.split(None, 4)
                    win_id = parts[0] if len(parts) > 0 else "?"
                    title = parts[4] if len(parts) > 4 else "untitled"
                    lines.append(f"  [{win_id}] {title}")
            return "\n".join(lines)
        except Exception as e:
            return f"Error listing windows: {e}"
    
    def focus_window(self, window_name: str) -> str:
        """Bring a window to focus by title (partial match)."""
        try:
            result = subprocess.run(
                ["wmctrl", "-a", window_name],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                return f"Focused window: {window_name}"
            return f"Window not found: {window_name}"
        except Exception as e:
            return f"Error: {e}"
    
    def close_window(self, window_name: str) -> str:
        """Close a window by title."""
        try:
            result = subprocess.run(
                ["wmctrl", "-c", window_name],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                return f"Closed window: {window_name}"
            return f"Window not found: {window_name}"
        except Exception as e:
            return f"Error: {e}"
    
    def run_command(self, command: str, timeout: int = 30) -> str:
        """Run a shell command with security controls."""
        
        # ── SECURITY LAYER ──────────────────────────────────────────────
        # Tier 1: Always allowed (read-only, safe)
        SAFE_PREFIXES = [
            "ls", "cat", "head", "tail", "wc", "grep", "find", "which",
            "whoami", "date", "uptime", "uname", "hostname", "df", "du",
            "free", "top -b -n 1", "ps", "id", "pwd", "echo", "file",
            "stat", "lsblk", "lscpu", "lsusb", "ip addr", "ip route",
            "ss -tulpn", "systemctl status", "journalctl", "dpkg -l",
            "apt list", "pip list", "pip show", "ollama list", "nvidia-smi",
            "xdg-open", "xdotool", "wmctrl", "xclip", "scrot",
            "firefox", "code", "nautilus", "thunar", "gedit", "nano",
            "python", "node", "cargo", "git status", "git log", "git diff",
            "git branch", "docker ps", "docker images",
        ]
        
        # Tier 2: Blocked (destructive / dangerous)
        BLOCKED_PATTERNS = [
            "rm -rf /", "rm -rf /*", "rm -rf ~", "rm -rf $HOME",
            "mkfs", "dd if=", ":(){ ", "fork bomb",
            "> /dev/sd", "chmod -R 777 /", "chown -R",
            "curl | sh", "curl | bash", "wget | sh", "wget | bash",
            "eval(", "exec(", "python -c 'import os; os.system",
            "/etc/passwd", "/etc/shadow", "passwd",
            "shutdown", "reboot", "poweroff", "init 0", "init 6",
            "iptables -F", "ufw disable",
            "nc -l", "ncat", "netcat",  # reverse shells
            "; rm ", "&& rm ", "| rm ",  # chained destructive
            "sudo su", "su -", "su root",
        ]
        
        cmd_lower = command.lower().strip()
        
        # Check blocked first
        for pattern in BLOCKED_PATTERNS:
            if pattern.lower() in cmd_lower:
                return f"🚫 BLOCKED: Command contains dangerous pattern '{pattern}'. This command was not executed."
        
        # Check if safe (starts with known safe command)
        cmd_base = cmd_lower.split()[0] if cmd_lower.split() else ""
        is_safe = any(
            cmd_lower.startswith(prefix) or cmd_base == prefix.split()[0]
            for prefix in SAFE_PREFIXES
        )
        
        # Tier 3: Unknown commands — allow but warn
        if not is_safe:
            # Allow but log a warning
            warning = f"⚠️  Unrecognized command pattern: '{cmd_base}'. Executing with caution.\n"
        else:
            warning = ""
        
        try:
            result = subprocess.run(
                command, shell=True,
                capture_output=True, text=True,
                timeout=timeout,
                env={**os.environ, "PATH": os.environ.get("PATH", "/usr/bin:/bin")},
            )
            output = warning + result.stdout
            if result.stderr:
                output += f"\nSTDERR: {result.stderr}"
            if result.returncode != 0:
                output += f"\nExit code: {result.returncode}"
            
            # Truncate if too long
            if len(output) > 5000:
                output = output[:5000] + "\n... [output truncated]"
            
            return output if output.strip() else "(no output)"
        except subprocess.TimeoutExpired:
            return f"Command timed out after {timeout}s"
        except Exception as e:
            return f"Error: {e}"
    
    def clipboard_read(self) -> str:
        """Read current clipboard contents."""
        try:
            result = subprocess.run(
                ["xclip", "-selection", "clipboard", "-o"],
                capture_output=True, text=True, timeout=5
            )
            content = result.stdout
            if not content:
                return "Clipboard is empty."
            if len(content) > 3000:
                content = content[:3000] + "\n... [truncated]"
            return f"Clipboard contents:\n{content}"
        except Exception as e:
            return f"Error reading clipboard: {e}"
    
    def clipboard_write(self, text: str) -> str:
        """Write text to the clipboard."""
        try:
            process = subprocess.Popen(
                ["xclip", "-selection", "clipboard"],
                stdin=subprocess.PIPE
            )
            process.communicate(text.encode())
            return f"Copied to clipboard ({len(text)} chars)"
        except Exception as e:
            return f"Error writing clipboard: {e}"
    
    def screenshot(self, region: str = "full") -> str:
        """Take a screenshot. Saves to ~/Screenshots/."""
        screenshots_dir = Path.home() / "Screenshots"
        screenshots_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = screenshots_dir / f"screenshot_{timestamp}.png"
        
        try:
            if region == "full":
                # Try different screenshot tools
                for tool_cmd in [
                    ["gnome-screenshot", "-f", str(filepath)],
                    ["scrot", str(filepath)],
                    ["import", "-window", "root", str(filepath)],  # ImageMagick
                ]:
                    if shutil.which(tool_cmd[0]):
                        subprocess.run(tool_cmd, timeout=10)
                        if filepath.exists():
                            return f"Screenshot saved: {filepath}"
                return "No screenshot tool found. Install: sudo apt install scrot"
            elif region == "window":
                if shutil.which("scrot"):
                    subprocess.run(["scrot", "-u", str(filepath)], timeout=10)
                    if filepath.exists():
                        return f"Window screenshot saved: {filepath}"
                return "scrot not available for window capture."
        except Exception as e:
            return f"Screenshot error: {e}"
    
    def send_keys(self, keys: str) -> str:
        """Send keyboard input using xdotool. Use xdotool key syntax."""
        try:
            subprocess.run(["xdotool", "key", keys], timeout=5)
            return f"Sent keys: {keys}"
        except Exception as e:
            return f"Error sending keys: {e}"
    
    def type_text(self, text: str) -> str:
        """Type text as keyboard input (like physically typing)."""
        try:
            subprocess.run(["xdotool", "type", "--delay", "20", text], timeout=30)
            return f"Typed: {text[:50]}{'...' if len(text) > 50 else ''}"
        except Exception as e:
            return f"Error typing: {e}"
    
    def get_active_window(self) -> str:
        """Get info about the currently focused window."""
        try:
            win_id = subprocess.run(
                ["xdotool", "getactivewindow"],
                capture_output=True, text=True, timeout=5
            ).stdout.strip()
            
            win_name = subprocess.run(
                ["xdotool", "getactivewindow", "getwindowname"],
                capture_output=True, text=True, timeout=5
            ).stdout.strip()
            
            win_pid = subprocess.run(
                ["xdotool", "getactivewindow", "getwindowpid"],
                capture_output=True, text=True, timeout=5
            ).stdout.strip()
            
            return f"Active window: {win_name} (ID: {win_id}, PID: {win_pid})"
        except Exception as e:
            return f"Error: {e}"
    
    # ── Tool definitions ────────────────────────────────────────────────────
    
    def get_tool_definitions(self) -> list[dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "launch_app",
                    "description": "Launch an application by name or command. Can also open URLs and files.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "app_name": {"type": "string", "description": "App name or command (e.g., 'firefox', 'code', 'nautilus ~/Documents')"}
                        },
                        "required": ["app_name"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "list_windows",
                    "description": "List all currently open windows with their IDs and titles.",
                    "parameters": {"type": "object", "properties": {}}
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "focus_window",
                    "description": "Bring a window to the foreground by its title (partial match).",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "window_name": {"type": "string", "description": "Window title to focus (partial match)"}
                        },
                        "required": ["window_name"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "run_command",
                    "description": "Execute a shell command and return the output. Use for system tasks, installing packages, running scripts, etc.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "command": {"type": "string", "description": "Shell command to run"},
                            "timeout": {"type": "integer", "description": "Timeout in seconds (default: 30)"}
                        },
                        "required": ["command"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "clipboard_read",
                    "description": "Read the current system clipboard contents.",
                    "parameters": {"type": "object", "properties": {}}
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "clipboard_write",
                    "description": "Write text to the system clipboard.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "text": {"type": "string", "description": "Text to copy to clipboard"}
                        },
                        "required": ["text"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "screenshot",
                    "description": "Take a screenshot of the screen or active window.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "region": {"type": "string", "enum": ["full", "window"], "description": "Capture full screen or active window"}
                        }
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "send_keys",
                    "description": "Send keyboard shortcuts (e.g., 'ctrl+c', 'alt+F4', 'Return'). Uses xdotool key syntax.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "keys": {"type": "string", "description": "Key combination in xdotool format"}
                        },
                        "required": ["keys"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "type_text",
                    "description": "Type text as if from a keyboard into the currently focused window.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "text": {"type": "string", "description": "Text to type"}
                        },
                        "required": ["text"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_active_window",
                    "description": "Get information about the currently focused/active window.",
                    "parameters": {"type": "object", "properties": {}}
                }
            }
        ]
    
    def get_handlers(self) -> dict[str, Callable]:
        return {
            "launch_app": self.launch_app,
            "list_windows": self.list_windows,
            "focus_window": self.focus_window,
            "close_window": self.close_window,
            "run_command": self.run_command,
            "clipboard_read": self.clipboard_read,
            "clipboard_write": self.clipboard_write,
            "screenshot": self.screenshot,
            "send_keys": self.send_keys,
            "type_text": self.type_text,
            "get_active_window": self.get_active_window,
        }