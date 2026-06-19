from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from config import LabConfig, load_config
from memory_store import estimate_tokens
from model_provider import build_chat_model


@dataclass
class SessionState:
    messages: list[dict[str, str]] = field(default_factory=list)
    token_usage: int = 0
    prompt_tokens_processed: int = 0


class BaselineAgent:
    """Student TODO: implement Agent A.

    Requirements:
    - Within-session memory only
    - No persistent `User.md`
    - Should forget long-term facts across new threads
    """

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.sessions: dict[str, SessionState] = {}

        # TODO: optionally initialize a real LangChain/LangGraph agent when dependencies exist.
        self.langchain_agent = None
        self._maybe_build_langchain_agent()

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        if self.force_offline or not self.langchain_agent:
            return self._reply_offline(thread_id, message)
        
        # Live path
        from langchain_core.messages import HumanMessage
        run_config = {"configurable": {"thread_id": thread_id}}
        
        if thread_id not in self.sessions:
            self.sessions[thread_id] = SessionState()
        session = self.sessions[thread_id]
        
        prompt_tokens = sum(estimate_tokens(m["content"]) for m in session.messages) + estimate_tokens(message)
        session.prompt_tokens_processed += prompt_tokens
        session.messages.append({"role": "user", "content": message})
        
        res = self.langchain_agent.invoke({"messages": [HumanMessage(content=message)]}, run_config)
        reply_text = res["messages"][-1].content
        
        out_tokens = estimate_tokens(reply_text)
        session.token_usage += out_tokens
        session.messages.append({"role": "assistant", "content": reply_text})
        
        return {
            "reply": reply_text,
            "tokens": out_tokens
        }

    def token_usage(self, thread_id: str) -> int:
        if thread_id in self.sessions:
            return self.sessions[thread_id].token_usage
        return 0

    def prompt_token_usage(self, thread_id: str) -> int:
        if thread_id in self.sessions:
            return self.sessions[thread_id].prompt_tokens_processed
        return 0

    def compaction_count(self, thread_id: str) -> int:
        # Baseline has no compact memory.
        return 0

    def _reply_offline(self, thread_id: str, message: str) -> dict[str, Any]:
        if thread_id not in self.sessions:
            self.sessions[thread_id] = SessionState()
            
        session = self.sessions[thread_id]
        
        # Calculate prompt token cost for this turn (all past messages + new message)
        prompt_tokens = sum(estimate_tokens(m["content"]) for m in session.messages) + estimate_tokens(message)
        session.prompt_tokens_processed += prompt_tokens
        
        # Append user message
        session.messages.append({"role": "user", "content": message})
        
        # Simple offline logic to match tests
        # Baseline doesn't have cross-session memory.
        # But we answer based on CURRENT thread content for short-term recall.
        
        # Simple search in current thread
        reply_text = "Tôi đã ghi nhận."
        lower_msg = message.lower()
        if "tên" in lower_msg and "gì" in lower_msg:
            # try to find in thread
            for m in reversed(session.messages):
                if m["role"] == "user" and "tên" in m["content"].lower() and "là" in m["content"].lower():
                    # rough heuristic
                    parts = m["content"].split("là")
                    if len(parts) > 1:
                        reply_text = f"Bạn tên là {parts[1].strip()}."
                        break
        
        # Token usage for output
        out_tokens = estimate_tokens(reply_text)
        session.token_usage += out_tokens
        
        # Append assistant message
        session.messages.append({"role": "assistant", "content": reply_text})
        
        return {
            "reply": reply_text,
            "tokens": out_tokens
        }

    def _maybe_build_langchain_agent(self):
        """Wire `create_agent` + `InMemorySaver` here."""
        if self.force_offline:
            return
            
        try:
            from langgraph.prebuilt import create_react_agent
            from langgraph.checkpoint.memory import MemorySaver
            
            self.chat_model = build_chat_model(self.config.model)
            self.memory = MemorySaver()
            self.langchain_agent = create_react_agent(self.chat_model, tools=[], checkpointer=self.memory)
        except ImportError:
            self.langchain_agent = None
