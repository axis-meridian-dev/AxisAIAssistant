"""
File Management Tool — full filesystem access for the local AI assistant.

Capabilities:
- List directory contents with metadata
- Read file contents (text files)
- Write/create files
- Search for files by name or content (using find/grep)
- Move, copy, rename files
- Get file info (size, permissions, modified date)
- Create directories
- Delete files (with safety checks)
"""

import os
import shutil
import subprocess
import stat
from pathlib import Path
from datetime import datetime
from typing import Callable

from tools.base import BaseTool


class FileManagerTool(BaseTool):
    
    def __init__(self, config: dict):
        super().__init__(config)
        self.home = Path.home()
        self.max_file_size = config.get("files", {}).get("max_file_size_mb", 50) * 1024 * 1024
        self.excluded = config.get("files", {}).get("excluded_dirs", [])
    
    def _resolve(self, path: str) -> Path:
        """Resolve path, expanding ~ and making absolute."""
        p = Path(path).expanduser()
        if not p.is_absolute():
            p = self.home / p
        return p.resolve()
    
    def _check_allowed(self, path: Path) -> bool:
        """Check if path is within allowed roots."""
        allowed = self.config.get("files", {}).get("allowed_roots", ["~"])
        for root in allowed:
            root_path = Path(root).expanduser().resolve()
            try:
                path.resolve().relative_to(root_path)
                return True
            except ValueError:
                continue
        return False
    
    def list_directory(self, path: str = "~", show_hidden: bool = False, 
                       sort_by: str = "name") -> str:
        """List contents of a directory with details."""
        target = self._resolve(path)
        
        if not target.exists():
            return f"Directory not found: {target}"
        if not target.is_dir():
            return f"Not a directory: {target}"
        
        try:
            entries = list(target.iterdir())
        except PermissionError:
            return f"Permission denied: {target}"
        
        if not show_hidden:
            entries = [e for e in entries if not e.name.startswith(".")]
        
        # Sort
        if sort_by == "size":
            entries.sort(key=lambda e: e.stat().st_size if e.is_file() else 0, reverse=True)
        elif sort_by == "modified":
            entries.sort(key=lambda e: e.stat().st_mtime, reverse=True)
        else:
            entries.sort(key=lambda e: (not e.is_dir(), e.name.lower()))
        
        lines = [f"📁 {target}\n"]
        for entry in entries:
            try:
                st = entry.stat()
                if entry.is_dir():
                    lines.append(f"  📂 {entry.name}/")
                else:
                    size = self._human_size(st.st_size)
                    mtime = datetime.fromtimestamp(st.st_mtime).strftime("%Y-%m-%d %H:%M")
                    lines.append(f"  📄 {entry.name}  ({size}, {mtime})")
            except (PermissionError, OSError):
                lines.append(f"  ⚠️  {entry.name}  (inaccessible)")
        
        lines.append(f"\n{len(entries)} items")
        return "\n".join(lines)
    
    def read_file(self, path: str, max_lines: int = 500) -> str:
        """Read contents of a text file."""
        target = self._resolve(path)
        
        if not target.exists():
            return f"File not found: {target}"
        if not target.is_file():
            return f"Not a file: {target}"
        if target.stat().st_size > self.max_file_size:
            return f"File too large ({self._human_size(target.stat().st_size)}). Max: {self.config['files']['max_file_size_mb']}MB"
        
        try:
            with open(target, "r", errors="replace") as f:
                lines = f.readlines()
            
            if len(lines) > max_lines:
                content = "".join(lines[:max_lines])
                content += f"\n\n... [{len(lines) - max_lines} more lines truncated]"
            else:
                content = "".join(lines)
            
            return f"File: {target} ({len(lines)} lines)\n{'─' * 40}\n{content}"
        except Exception as e:
            return f"Error reading file: {e}"
    
    def write_file(self, path: str, content: str, append: bool = False) -> str:
        """Write content to a file. Creates parent dirs if needed."""
        target = self._resolve(path)
        
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            mode = "a" if append else "w"
            with open(target, mode) as f:
                f.write(content)
            return f"{'Appended to' if append else 'Written'}: {target} ({len(content)} chars)"
        except Exception as e:
            return f"Error writing file: {e}"
    
    def search_files(self, query: str, path: str = "~", 
                     search_type: str = "name", max_results: int = 20) -> str:
        """Search for files by name pattern or content."""
        target = self._resolve(path)
        
        exclude_args = []
        for d in self.excluded:
            exclude_args.extend(["--exclude-dir", d])
        
        try:
            if search_type == "content":
                # Search file contents with grep
                cmd = [
                    "grep", "-rl", "--include=*.{py,js,ts,json,md,txt,cfg,yml,yaml,toml,sh,html,css}",
                    "-i", *exclude_args, query, str(target)
                ]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                files = result.stdout.strip().split("\n")[:max_results]
                files = [f for f in files if f]
                
                if not files:
                    return f"No files containing '{query}' found in {target}"
                
                lines = [f"Files containing '{query}':\n"]
                for f in files:
                    lines.append(f"  📄 {f}")
                return "\n".join(lines)
            
            else:
                # Search by filename with find
                cmd = [
                    "find", str(target), "-maxdepth", "6",
                    "-iname", f"*{query}*", "-not", "-path", "*/.git/*",
                    "-not", "-path", "*/node_modules/*"
                ]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                files = result.stdout.strip().split("\n")[:max_results]
                files = [f for f in files if f]
                
                if not files:
                    return f"No files matching '*{query}*' found in {target}"
                
                lines = [f"Files matching '*{query}*':\n"]
                for f in files:
                    p = Path(f)
                    try:
                        size = self._human_size(p.stat().st_size) if p.is_file() else "dir"
                        lines.append(f"  {'📂' if p.is_dir() else '📄'} {f}  ({size})")
                    except OSError:
                        lines.append(f"  📄 {f}")
                return "\n".join(lines)
                
        except subprocess.TimeoutExpired:
            return "Search timed out. Try a more specific path or query."
        except Exception as e:
            return f"Search error: {e}"
    
    def move_file(self, source: str, destination: str) -> str:
        """Move or rename a file/directory."""
        src = self._resolve(source)
        dst = self._resolve(destination)
        
        if not src.exists():
            return f"Source not found: {src}"
        
        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst))
            return f"Moved: {src} → {dst}"
        except Exception as e:
            return f"Error moving: {e}"
    
    def copy_file(self, source: str, destination: str) -> str:
        """Copy a file or directory."""
        src = self._resolve(source)
        dst = self._resolve(destination)
        
        if not src.exists():
            return f"Source not found: {src}"
        
        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            if src.is_dir():
                shutil.copytree(str(src), str(dst))
            else:
                shutil.copy2(str(src), str(dst))
            return f"Copied: {src} → {dst}"
        except Exception as e:
            return f"Error copying: {e}"
    
    def delete_file(self, path: str) -> str:
        """Delete a file or empty directory. Returns confirmation message."""
        target = self._resolve(path)
        
        if not target.exists():
            return f"Not found: {target}"
        
        try:
            if target.is_dir():
                if any(target.iterdir()):
                    count = sum(1 for _ in target.rglob("*"))
                    return f"Directory not empty ({count} items). Use delete_directory for recursive deletion."
                target.rmdir()
            else:
                size = self._human_size(target.stat().st_size)
                target.unlink()
                return f"Deleted: {target} ({size})"
            return f"Deleted: {target}"
        except Exception as e:
            return f"Error deleting: {e}"
    
    def make_directory(self, path: str) -> str:
        """Create a directory (and parents if needed)."""
        target = self._resolve(path)
        try:
            target.mkdir(parents=True, exist_ok=True)
            return f"Created directory: {target}"
        except Exception as e:
            return f"Error creating directory: {e}"
    
    def file_info(self, path: str) -> str:
        """Get detailed info about a file or directory."""
        target = self._resolve(path)
        
        if not target.exists():
            return f"Not found: {target}"
        
        try:
            st = target.stat()
            info = {
                "path": str(target),
                "type": "directory" if target.is_dir() else "file",
                "size": self._human_size(st.st_size),
                "size_bytes": st.st_size,
                "permissions": stat.filemode(st.st_mode),
                "owner_uid": st.st_uid,
                "group_gid": st.st_gid,
                "modified": datetime.fromtimestamp(st.st_mtime).isoformat(),
                "created": datetime.fromtimestamp(st.st_ctime).isoformat(),
                "accessed": datetime.fromtimestamp(st.st_atime).isoformat(),
            }
            
            if target.is_dir():
                info["items"] = sum(1 for _ in target.iterdir())
            elif target.is_file():
                info["extension"] = target.suffix
                # Try to detect if it's a text file
                try:
                    with open(target, "r") as f:
                        f.read(512)
                    info["text_file"] = True
                    info["lines"] = sum(1 for _ in open(target))
                except (UnicodeDecodeError, IsADirectoryError):
                    info["text_file"] = False
            
            return "\n".join(f"  {k}: {v}" for k, v in info.items())
        except Exception as e:
            return f"Error getting info: {e}"
    
    def disk_usage(self, path: str = "~") -> str:
        """Get disk usage for a path."""
        target = self._resolve(path)
        try:
            result = subprocess.run(
                ["du", "-sh", str(target)],
                capture_output=True, text=True, timeout=30
            )
            usage = result.stdout.strip()
            
            # Also get filesystem info
            total, used, free = shutil.disk_usage(target)
            return (
                f"Directory size: {usage.split()[0]}\n"
                f"Filesystem: {self._human_size(total)} total, "
                f"{self._human_size(used)} used, "
                f"{self._human_size(free)} free "
                f"({int(used/total*100)}% used)"
            )
        except Exception as e:
            return f"Error: {e}"
    
    def _human_size(self, size: int) -> str:
        for unit in ["B", "KB", "MB", "GB", "TB"]:
            if abs(size) < 1024:
                return f"{size:.1f}{unit}"
            size /= 1024
        return f"{size:.1f}PB"
    
    # ── Tool definition for Ollama ──────────────────────────────────────────
    
    def get_tool_definitions(self) -> list[dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "list_directory",
                    "description": "List files and directories at a given path. Returns names, sizes, and modification dates.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "Directory path (default: ~)"},
                            "show_hidden": {"type": "boolean", "description": "Show hidden files (default: false)"},
                            "sort_by": {"type": "string", "enum": ["name", "size", "modified"], "description": "Sort order"}
                        }
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read the text contents of a file. Use for viewing code, configs, documents, etc.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "File path to read"},
                            "max_lines": {"type": "integer", "description": "Max lines to read (default: 500)"}
                        },
                        "required": ["path"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "write_file",
                    "description": "Write content to a file. Creates the file if it doesn't exist. Creates parent directories as needed.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "File path to write to"},
                            "content": {"type": "string", "description": "Content to write"},
                            "append": {"type": "boolean", "description": "Append instead of overwrite (default: false)"}
                        },
                        "required": ["path", "content"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "search_files",
                    "description": "Search for files by name pattern or by content (grep). Use to find files on the system.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Search query (filename pattern or content text)"},
                            "path": {"type": "string", "description": "Directory to search in (default: ~)"},
                            "search_type": {"type": "string", "enum": ["name", "content"], "description": "Search by filename or file content"},
                            "max_results": {"type": "integer", "description": "Max results to return (default: 20)"}
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "move_file",
                    "description": "Move or rename a file or directory.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "source": {"type": "string", "description": "Source path"},
                            "destination": {"type": "string", "description": "Destination path"}
                        },
                        "required": ["source", "destination"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "copy_file",
                    "description": "Copy a file or directory to a new location.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "source": {"type": "string", "description": "Source path"},
                            "destination": {"type": "string", "description": "Destination path"}
                        },
                        "required": ["source", "destination"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "delete_file",
                    "description": "Delete a file or empty directory. Will refuse to delete non-empty directories for safety.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "Path to delete"}
                        },
                        "required": ["path"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "make_directory",
                    "description": "Create a new directory (and any parent directories needed).",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "Directory path to create"}
                        },
                        "required": ["path"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "file_info",
                    "description": "Get detailed metadata about a file or directory (size, permissions, dates, type).",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "Path to inspect"}
                        },
                        "required": ["path"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "disk_usage",
                    "description": "Get disk usage for a directory and overall filesystem stats.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "Path to check (default: ~)"}
                        }
                    }
                }
            }
        ]
    
    def get_handlers(self) -> dict[str, Callable]:
        return {
            "list_directory": self.list_directory,
            "read_file": self.read_file,
            "write_file": self.write_file,
            "search_files": self.search_files,
            "move_file": self.move_file,
            "copy_file": self.copy_file,
            "delete_file": self.delete_file,
            "make_directory": self.make_directory,
            "file_info": self.file_info,
            "disk_usage": self.disk_usage,
        }
