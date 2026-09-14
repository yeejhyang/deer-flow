import asyncio
from typing import Any

import pytest
from langgraph.checkpoint.base import BaseCheckpointSaver, CheckpointTuple

from deerflow.runtime.checkpoint_cache.base import CheckpointCacheStats, make_history_key, thread_key_stem
from deerflow.runtime.checkpointer.cached_saver import CachedHistorySaver

PREFIX = "ckpt-hist:v1:cancel-test"


def _cfg(thread_id: str, checkpoint_id: str) -> dict[str, Any]:
    return {
        "configurable": {
            "thread_id": thread_id,
            "checkpoint_ns": "",
            "checkpoint_id": checkpoint_id,
        }
    }


def _tuple(thread_id: str, checkpoint_id: str, parent_id: str | None, *, seed: str | None = None) -> CheckpointTuple:
    parent = _cfg(thread_id, parent_id) if parent_id is not None else None
    values = {"messages": [seed]} if seed is not None else {}
    checkpoint = {
        "v": 1,
        "id": checkpoint_id,
        "channel_values": values,
        "channel_versions": {},
        "versions_seen": {},
        "updated_at": None,
    }
    return CheckpointTuple(
        config=_cfg(thread_id, checkpoint_id),
        checkpoint=checkpoint,
        metadata={},
        parent_config=parent,
        pending_writes=[],
    )


class _PruningSaver(BaseCheckpointSaver):
    """Keeps the latest checkpoint id but rewrites its parent chain on prune."""

    def __init__(self) -> None:
        super().__init__()
        self._tuples = {
            "latest": _tuple("t1", "latest", "old-parent"),
            "old-parent": _tuple("t1", "old-parent", None, seed="old-seed"),
        }
        self.pruned = asyncio.Event()

    def get_tuple(self, config):
        return self._tuples.get(config["configurable"].get("checkpoint_id"))

    async def aget_tuple(self, config):
        return self.get_tuple(config)

    def list(self, config, *, filter=None, before=None, limit=None):
        yield from ()

    def put(self, config, checkpoint, metadata, new_versions):
        raise NotImplementedError

    def put_writes(self, config, writes, task_id, task_path=""):
        raise NotImplementedError

    def delete_thread(self, thread_id):
        raise NotImplementedError

    async def aprune(self, thread_ids, *, strategy="keep_latest"):
        assert list(thread_ids) == ["t1"]
        assert strategy == "keep_latest"
        self._tuples = {
            "latest": _tuple("t1", "latest", "new-parent"),
            "new-parent": _tuple("t1", "new-parent", None, seed="new-seed"),
        }
        self.pruned.set()


class _BlockingPurgeCache:
    """Redis-shaped async cache whose thread purge has a cancellable await."""

    enabled = True

    def __init__(self) -> None:
        key = make_history_key(PREFIX, "t1", "", "latest", "messages")
        self.data = {key: {"writes": [], "seed": ["old-seed"]}}
        self.purge_started = asyncio.Event()
        self.release_purge = asyncio.Event()

    async def aget_many(self, keys):
        return {key: self.data[key] for key in keys if key in self.data}

    async def aset_many(self, entries):
        self.data.update(entries)

    async def adelete_thread(self, key_prefix: str, thread_id: str) -> None:
        self.purge_started.set()
        await self.release_purge.wait()
        stem = thread_key_stem(key_prefix, thread_id)
        for key in [key for key in self.data if key.startswith(stem)]:
            del self.data[key]

    def stats(self) -> CheckpointCacheStats:
        return CheckpointCacheStats(entries=len(self.data))


@pytest.mark.anyio
async def test_aprune_cancellation_does_not_leave_pre_prune_history_cached():
    inner = _PruningSaver()
    cache = _BlockingPurgeCache()
    saver = CachedHistorySaver(inner, cache, key_prefix=PREFIX)

    before = await saver.aget_delta_channel_history(config=_cfg("t1", "latest"), channels=["messages"])
    assert before["messages"]["seed"] == ["old-seed"]

    pruning = asyncio.create_task(saver.aprune(["t1"], strategy="keep_latest"))
    await inner.pruned.wait()
    await cache.purge_started.wait()

    pruning.cancel()
    await asyncio.sleep(0)

    # The source-of-truth chain has already changed. Cancellation must not let
    # aprune finish until the mandatory cache purge settles.
    assert not pruning.done()

    # Repeated cancellation must not detach the purge either.
    pruning.cancel()
    await asyncio.sleep(0)
    assert not pruning.done()

    cache.release_purge.set()
    with pytest.raises(asyncio.CancelledError):
        await pruning

    after = await saver.aget_delta_channel_history(config=_cfg("t1", "latest"), channels=["messages"])
    assert after["messages"]["seed"] == ["new-seed"]
