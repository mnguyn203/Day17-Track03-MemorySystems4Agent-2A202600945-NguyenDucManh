from __future__ import annotations

from pathlib import Path

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import load_config


def make_config(tmp_path: Path):
    """Build an isolated config for tests."""
    from model_provider import ProviderConfig
    return load_config().replace(
        state_dir=tmp_path,
        compact_threshold_tokens=20,
        compact_keep_messages=2
    ) if hasattr(load_config(), 'replace') else None


def test_user_markdown_read_write_edit(tmp_path: Path) -> None:
    """Verify `User.md` can be created, updated, and edited."""
    from memory_store import UserProfileStore
    store = UserProfileStore(tmp_path)
    user_id = "test_user_123"
    
    assert store.read_text(user_id) == ""
    
    store.write_text(user_id, "- name: John")
    assert "John" in store.read_text(user_id)
    
    changed = store.edit_text(user_id, "John", "Jane")
    assert changed is True
    assert "Jane" in store.read_text(user_id)
    assert store.file_size(user_id) > 0


def test_compact_trigger(tmp_path: Path) -> None:
    """Verify long threads trigger compaction."""
    from config import LabConfig
    from model_provider import ProviderConfig
    config = LabConfig(
        base_dir=tmp_path,
        data_dir=tmp_path,
        state_dir=tmp_path,
        compact_threshold_tokens=20,
        compact_keep_messages=2,
        model=ProviderConfig("openai", "gpt", 0.0),
        judge_model=ProviderConfig("openai", "gpt", 0.0)
    )
    
    agent = AdvancedAgent(config=config, force_offline=True)
    user_id = "u1"
    thread_id = "t1"
    
    for i in range(5):
        agent.reply(user_id, thread_id, f"Câu hỏi số {i} " * 5)
        
    assert agent.compaction_count(thread_id) > 0


def test_cross_session_recall(tmp_path: Path) -> None:
    """Verify advanced remembers across sessions and baseline does not."""
    from config import LabConfig
    from model_provider import ProviderConfig
    config = LabConfig(
        base_dir=tmp_path,
        data_dir=tmp_path,
        state_dir=tmp_path,
        compact_threshold_tokens=200,
        compact_keep_messages=5,
        model=ProviderConfig("openai", "gpt", 0.0),
        judge_model=ProviderConfig("openai", "gpt", 0.0)
    )
    
    adv_agent = AdvancedAgent(config=config, force_offline=True)
    base_agent = BaselineAgent(config=config, force_offline=True)
    
    adv_agent.reply("u1", "t1", "tên tôi là Dũng")
    base_agent.reply("u1", "t1", "tên tôi là Dũng")
    
    adv_reply = adv_agent.reply("u1", "t2", "tôi tên là gì?")
    base_reply = base_agent.reply("u1", "t2", "tôi tên là gì?")
    
    assert "dũng" in adv_reply["reply"].lower()
    assert "dũng" not in base_reply["reply"].lower()


def test_compact_reduces_prompt_load_on_long_thread(tmp_path: Path) -> None:
    """Compare prompt load of baseline vs advanced on a long thread."""
    from config import LabConfig
    from model_provider import ProviderConfig
    config = LabConfig(
        base_dir=tmp_path,
        data_dir=tmp_path,
        state_dir=tmp_path,
        compact_threshold_tokens=50,
        compact_keep_messages=2,
        model=ProviderConfig("openai", "gpt", 0.0),
        judge_model=ProviderConfig("openai", "gpt", 0.0)
    )
    
    adv_agent = AdvancedAgent(config=config, force_offline=True)
    base_agent = BaselineAgent(config=config, force_offline=True)
    
    for i in range(10):
        msg = f"Nói cho tôi biết về điều số {i} với thật nhiều chi tiết. " * 5
        adv_agent.reply("u1", "t1", msg)
        base_agent.reply("u1", "t1", msg)
        
    assert adv_agent.prompt_token_usage("t1") < base_agent.prompt_token_usage("t1")
