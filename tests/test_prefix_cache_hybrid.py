# SPDX-License-Identifier: Apache-2.0
"""Tests for block-aware hybrid prefix cache functionality."""

from unittest.mock import MagicMock

from vllm_mlx.paged_cache import PagedCacheManager
from vllm_mlx.prefix_cache import BlockAwarePrefixCache, PrefixCacheManager


class TestPrefixCacheManagerHybrid:
    """Small regression tests for PrefixCacheManager compatibility."""

    def test_store_and_fetch_cache(self):
        mock_model = MagicMock()
        cache_manager = PrefixCacheManager(model=mock_model, max_entries=10)

        tokens = list(range(50))
        fake_cache = ["layer0", "layer1"]

        cache_manager.store_cache(tokens, fake_cache)
        result, remaining = cache_manager.fetch_cache(tokens)

        assert result is not None
        assert remaining == []

    def test_shorter_prefix_match(self):
        mock_model = MagicMock()
        cache_manager = PrefixCacheManager(model=mock_model, max_entries=10)

        tokens = list(range(50))
        fake_cache = ["layer0", "layer1"]

        cache_manager.store_cache(tokens, fake_cache)

        longer_tokens = list(range(70))
        result, remaining = cache_manager.fetch_cache(longer_tokens)

        assert result is not None
        assert remaining == list(range(50, 70))


class TestBlockAwarePrefixCache:
    """Tests for BlockAwarePrefixCache."""

    def test_imports(self):
        assert BlockAwarePrefixCache is not None
        assert PrefixCacheManager is not None

    def test_initialization(self):
        paged_manager = PagedCacheManager(block_size=64, max_blocks=100)
        cache = BlockAwarePrefixCache(model=None, paged_cache_manager=paged_manager)

        assert cache.block_size == 64
        assert len(cache) == 0

    def test_store_and_fetch_cache(self):
        paged_manager = PagedCacheManager(block_size=64, max_blocks=100)
        cache = BlockAwarePrefixCache(model=None, paged_cache_manager=paged_manager)

        tokens = list(range(128))
        block_table = cache.store_cache("req-1", tokens, ["cache_data_1"])

        assert block_table is not None
        assert block_table.num_tokens == 128
        assert len(block_table.block_ids) == 2

        block_table2, remaining = cache.fetch_cache("req-2", tokens + [999, 1000])

        assert block_table2 is not None
        assert remaining == [999, 1000]

    def test_release_cache(self):
        paged_manager = PagedCacheManager(block_size=64, max_blocks=100)
        cache = BlockAwarePrefixCache(model=None, paged_cache_manager=paged_manager)

        cache.store_cache("req-1", list(range(64)), ["data"])
        assert len(cache) == 1

        cache.release_cache("req-1")
        assert len(cache) == 0

    def test_fork_cache(self):
        paged_manager = PagedCacheManager(block_size=64, max_blocks=100)
        cache = BlockAwarePrefixCache(model=None, paged_cache_manager=paged_manager)

        cache.store_cache("req-1", list(range(128)), ["shared_data"])
        forked_table = cache.fork_cache("req-1", "req-2")

        assert forked_table is not None
        assert len(cache) == 2
        assert cache.get_stats()["shared_blocks"] > 0

    def test_get_cache_for_generation(self):
        paged_manager = PagedCacheManager(block_size=64, max_blocks=100)
        cache = BlockAwarePrefixCache(model=None, paged_cache_manager=paged_manager)

        cache.store_cache("req-1", list(range(64)), ["data"])
        cache_data, was_copied = cache.get_cache_for_generation("req-1")

        assert cache_data == ["data"]
        assert was_copied is False

    def test_get_cache_for_generation_with_cow(self):
        paged_manager = PagedCacheManager(block_size=64, max_blocks=100)
        cache = BlockAwarePrefixCache(model=None, paged_cache_manager=paged_manager)

        cache.store_cache("req-1", list(range(64)), ["shared_data"])
        cache.fork_cache("req-1", "req-2")

        cache_data, was_copied = cache.get_cache_for_generation("req-2")

        assert cache_data is not None
        assert was_copied is True

    def test_stats(self):
        paged_manager = PagedCacheManager(block_size=64, max_blocks=100)
        cache = BlockAwarePrefixCache(model=None, paged_cache_manager=paged_manager)

        cache.fetch_cache("req-1", [1, 2, 3])

        stats = cache.get_stats()
        assert stats["misses"] == 1
        assert stats["hits"] == 0

    def test_clear(self):
        paged_manager = PagedCacheManager(block_size=64, max_blocks=100)
        cache = BlockAwarePrefixCache(model=None, paged_cache_manager=paged_manager)

        tokens = list(range(128))
        cache.store_cache("req-1", tokens, ["data"])
        cache.store_cache("req-2", tokens, ["data2"])

        assert len(cache) == 2

        cache.clear()

        assert len(cache) == 0
        assert cache.get_stats()["allocated_blocks"] == 1