"""跨页面共享的日志环形缓冲区

模块级单例，供 process_runner 推送原始 ANSI 日志，
LogViewer 和 /console 页面消费。
"""

import threading
from collections import deque
from typing import Callable, Dict, List, Tuple


class LogBuffer:
    """线程安全的日志环形缓冲区"""

    def __init__(self, maxlen: int = 5000):
        self._buf: deque[Tuple[int, str]] = deque(maxlen=maxlen)
        self._seq = 0
        self._lock = threading.Lock()
        self._subscribers: Dict[int, Callable[[int, str], None]] = {}
        self._next_sub_id = 0

    def push(self, line: str):
        """追加一行日志（原始 ANSI）并通知所有订阅者"""
        with self._lock:
            self._seq += 1
            seq = self._seq
            self._buf.append((seq, line))
            # 复制一份订阅者列表避免在回调中修改
            subs = list(self._subscribers.values())
        for cb in subs:
            try:
                cb(seq, line)
            except Exception:
                pass

    def subscribe(self, callback: Callable[[int, str], None]) -> int:
        """注册订阅者，返回订阅 ID

        callback 签名: (seq: int, line: str) -> None
        """
        with self._lock:
            sub_id = self._next_sub_id
            self._next_sub_id += 1
            self._subscribers[sub_id] = callback
        return sub_id

    def unsubscribe(self, sub_id: int):
        """取消订阅"""
        with self._lock:
            self._subscribers.pop(sub_id, None)

    def get_all_lines(self) -> List[Tuple[int, str]]:
        """获取缓冲区中所有行（用于新页面回放历史）"""
        with self._lock:
            return list(self._buf)

    def clear(self):
        """清空缓冲区（新进程启动时调用）"""
        with self._lock:
            self._buf.clear()
            self._seq = 0


# 模块级全局单例
log_buffer = LogBuffer()
