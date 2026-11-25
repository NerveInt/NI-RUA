"""跨协议 Egress 脱敏投影有界 LRU 缓存。"""

from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
import threading

_MISSING = object()


def _clone_json(value):
    """递归拷贝可变 JSON 容器，共享不可变标量与字符串。"""
    if isinstance(value, dict):
        return {key: _clone_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_json(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return deepcopy(value)


class _CachedPresentationProxy:
    """轻量代理：仅覆写 present_egress_text 走字符串 LRU，其余属性与方法原样透传。"""

    def __init__(self, target, cache: EgressProjectionCache):
        self._target = target
        self._cache = cache

    def present_egress_text(self, text):
        transform = getattr(self._target, "present_egress_text", self._target.present_text)
        if not isinstance(text, str):
            return transform(text)
        return self._cache.cached_text(text, transform)

    def __getattr__(self, name):
        return getattr(self._target, name)


class EgressProjectionCache:
    """线程安全的跨 Wire 有界 LRU 缓存，覆盖文本脱敏、工具目录与结构化条目投影。"""

    def __init__(
        self,
        max_text_entries: int = 2048,
        max_text_bytes: int = 8 * 1024 * 1024,
        max_single_text_bytes: int = 256 * 1024,
        max_item_entries: int = 512,
    ):
        self._max_text_entries = max(0, int(max_text_entries))
        self._max_text_bytes = max(0, int(max_text_bytes))
        self._max_single_text_bytes = max(0, int(max_single_text_bytes))
        self._max_item_entries = max(0, int(max_item_entries))
        self._lock = threading.Lock()
        self._text_cache: OrderedDict[str, tuple[str, int]] = OrderedDict()
        self._text_bytes = 0
        self._buckets: dict[str, OrderedDict] = {}

    @property
    def text_entries(self) -> int:
        with self._lock:
            return len(self._text_cache)

    @property
    def text_bytes(self) -> int:
        with self._lock:
            return self._text_bytes

    def bucket_size(self, bucket: str) -> int:
        with self._lock:
            store = self._buckets.get(bucket)
            return len(store) if store is not None else 0

    def wrap_presentation(self, presentation):
        """包装 ActorPresentation，使其 present_egress_text 自动复用字符串级 LRU。"""
        if presentation is None:
            return None
        return _CachedPresentationProxy(presentation, self)

    def cached_text(self, text: str, transform):
        """字符串到字符串的 LRU 缓存；转换函数在锁外执行，超单条或总字节上限则跳过缓存。"""
        if self._max_text_entries <= 0 or self._max_text_bytes <= 0:
            return transform(text)
        in_bytes = len(text.encode("utf-8"))
        if in_bytes > self._max_single_text_bytes or in_bytes > self._max_text_bytes:
            return transform(text)
        with self._lock:
            hit = self._text_cache.get(text)
            if hit is not None:
                self._text_cache.move_to_end(text)
                return hit[0]
        projected = transform(text)
        if not isinstance(projected, str):
            return projected
        entry_bytes = max(in_bytes, len(projected.encode("utf-8")))
        if entry_bytes > self._max_single_text_bytes or entry_bytes > self._max_text_bytes:
            return projected
        with self._lock:
            existing = self._text_cache.get(text)
            if existing is not None:
                self._text_cache.move_to_end(text)
                return existing[0]
            self._text_cache[text] = (projected, entry_bytes)
            self._text_bytes += entry_bytes
            while len(self._text_cache) > self._max_text_entries or self._text_bytes > self._max_text_bytes:
                _, (_, evicted_bytes) = self._text_cache.popitem(last=False)
                self._text_bytes -= evicted_bytes
        return projected

    def cached_item(self, bucket: str, key, transform):
        """按分桶执行结构化投影 LRU，返回隔离可变副本以防调用方原地修改污染缓存。"""
        if self._max_item_entries <= 0:
            return transform()
        cached = _MISSING
        with self._lock:
            store = self._buckets.get(bucket)
            if store is not None and key in store:
                store.move_to_end(key)
                cached = store[key]
        if cached is not _MISSING:
            return _clone_json(cached)
        projected = transform()
        stored = _clone_json(projected)
        existing = _MISSING
        with self._lock:
            store = self._buckets.setdefault(bucket, OrderedDict())
            if key in store:
                store.move_to_end(key)
                existing = store[key]
            else:
                store[key] = stored
                while len(store) > self._max_item_entries:
                    store.popitem(last=False)
        return _clone_json(existing) if existing is not _MISSING else projected

    def cached_tools(self, key, transform):
        """缓存顶层 tools 列表投影结果，返回隔离副本。"""
        return self.cached_item("tools", key, transform)
