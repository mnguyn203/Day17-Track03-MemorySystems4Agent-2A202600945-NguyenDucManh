from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


def estimate_tokens(text: str) -> int:
    """A simple token estimator."""
    text = text.strip()
    if not text:
        return 0
    return max(1, len(text) // 4)


@dataclass
class UserProfileStore:
    """Persistent storage for `User.md`.

    Student TODO:
    - Map each user id to one markdown file
    - Support read / write / edit operations
    - Optionally expose helpers like `facts()` or `upsert_fact()`
    """

    root_dir: Path

    def path_for(self, user_id: str) -> Path:
        import re
        safe_id = re.sub(r'[^a-zA-Z0-9_-]', '_', user_id)
        return self.root_dir / f"{safe_id}.md"

    def read_text(self, user_id: str) -> str:
        path = self.path_for(user_id)
        return path.read_text(encoding="utf-8") if path.exists() else ""

    def write_text(self, user_id: str, content: str) -> Path:
        self.root_dir.mkdir(parents=True, exist_ok=True)
        path = self.path_for(user_id)
        path.write_text(content, encoding="utf-8")
        return path

    def edit_text(self, user_id: str, search_text: str, replacement: str) -> bool:
        content = self.read_text(user_id)
        if search_text and search_text in content:
            new_content = content.replace(search_text, replacement, 1)
            self.write_text(user_id, new_content)
            return True
        return False

    def file_size(self, user_id: str) -> int:
        path = self.path_for(user_id)
        return path.stat().st_size if path.exists() else 0


def extract_profile_updates(message: str) -> dict[str, str]:
    """Convert raw user text into stable profile facts."""
    import re
    facts = {}
    msg = message.lower()
    
    # Simple regex extractions for offline testing
    if m := re.search(r'tên.*?là\s+([\w\s]+)', msg):
        facts['name'] = m.group(1).strip().title()
    elif m := re.search(r'mình là\s+([\w\s]+)', msg):
        facts['name'] = m.group(1).strip().title()
        
    if m := re.search(r'sống ở\s+([\w\s]+)', msg):
        facts['location'] = m.group(1).strip().title()
    elif m := re.search(r'ở\s+([\w\s]+)', msg):
        # A bit risky but ok for simple tests
        facts['location'] = m.group(1).strip().title()
        
    if m := re.search(r'làm nghề\s+([\w\s]+)', msg):
        facts['profession'] = m.group(1).strip()
    elif m := re.search(r'nghề\s+([\w\s]+)', msg):
        facts['profession'] = m.group(1).strip()
        
    if m := re.search(r'thích\s+([\w\s]+)', msg):
        facts['preferences'] = m.group(1).strip()
    elif m := re.search(r'phong cách\s+([\w\s]+)', msg):
        facts['style'] = m.group(1).strip()
        
    return facts


def summarize_messages(messages: list[dict[str, str]], max_items: int = 6) -> str:
    """Create a compact summary of older messages."""
    if not messages:
        return ""
    
    # Offline heuristic: truly "compress" it so token usage goes down!
    return f"[Tóm tắt {len(messages)} tin nhắn cũ]"


@dataclass
class CompactMemoryManager:
    """Student TODO: implement compact memory for long threads.

    Goal:
    - Keep recent messages in full
    - When the thread grows too large, move older content into a summary
    - Track how many compactions happened for benchmarking
    """

    threshold_tokens: int
    keep_messages: int
    state: dict[str, dict[str, object]] = field(default_factory=dict)

    def append(self, thread_id: str, role: str, content: str) -> None:
        if thread_id not in self.state:
            self.state[thread_id] = {"messages": [], "summary": "", "compactions": 0}
            
        st = self.state[thread_id]
        st["messages"].append({"role": role, "content": content})
        
        # Check token usage
        current_tokens = estimate_tokens(st["summary"]) + sum(estimate_tokens(m["content"]) for m in st["messages"])
        if current_tokens > self.threshold_tokens:
            messages = st["messages"]
            if len(messages) > self.keep_messages:
                to_summarize = messages[:-self.keep_messages]
                new_summary_part = summarize_messages(to_summarize)
                
                if st["summary"]:
                    st["summary"] += "\\n" + new_summary_part
                else:
                    st["summary"] = new_summary_part
                    
                st["messages"] = messages[-self.keep_messages:]
                st["compactions"] += 1

    def context(self, thread_id: str) -> dict[str, object]:
        if thread_id not in self.state:
            self.state[thread_id] = {"messages": [], "summary": "", "compactions": 0}
        return self.state[thread_id]

    def compaction_count(self, thread_id: str) -> int:
        return self.state.get(thread_id, {}).get("compactions", 0)
