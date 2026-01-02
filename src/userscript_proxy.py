import asyncio
import time
import uuid
from collections import deque
from dataclasses import dataclass
from typing import Any, AsyncIterator, Deque, Dict, List, Optional


@dataclass
class ProxyJob:
    job_id: str
    payload: dict
    created_at: float
    expires_at: float
    queue: asyncio.Queue
    claimed: bool = False
    done: bool = False
    error: Optional[str] = None


class ProxyJobQueue:
    def __init__(self) -> None:
        self._pending: Deque[str] = deque()
        self._jobs: Dict[str, ProxyJob] = {}
        self._lock = asyncio.Lock()
        self._new_job_event = asyncio.Event()
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def _ensure_loop(self) -> None:
        """Ensure asyncio primitives are bound to the current running loop.

        Pytest's `IsolatedAsyncioTestCase` uses a fresh event loop per test, so
        any module-level `asyncio.Lock`/`asyncio.Event`/`asyncio.Queue` objects
        must be recreated when the loop changes.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        if self._loop is loop:
            return
        self._loop = loop
        self._pending = deque()
        self._jobs = {}
        self._lock = asyncio.Lock()
        self._new_job_event = asyncio.Event()

    async def create_job(self, payload: dict, ttl_seconds: int = 60) -> ProxyJob:
        self._ensure_loop()
        ttl_seconds = max(5, min(int(ttl_seconds), 600))
        now = time.time()
        job = ProxyJob(
            job_id=uuid.uuid4().hex,
            payload=payload,
            created_at=now,
            expires_at=now + ttl_seconds,
            queue=asyncio.Queue(),
        )
        async with self._lock:
            self._jobs[job.job_id] = job
            self._pending.append(job.job_id)
            self._new_job_event.set()
        return job

    async def claim_job(self, timeout_seconds: float = 0.0) -> Optional[ProxyJob]:
        self._ensure_loop()
        deadline = time.time() + max(0.0, timeout_seconds)
        while True:
            async with self._lock:
                self._prune_locked()
                while self._pending:
                    job_id = self._pending.popleft()
                    job = self._jobs.get(job_id)
                    if not job or job.done or job.claimed:
                        continue
                    job.claimed = True
                    return job

            if timeout_seconds <= 0:
                return None
            wait_seconds = deadline - time.time()
            if wait_seconds <= 0:
                return None
            self._new_job_event.clear()
            try:
                await asyncio.wait_for(self._new_job_event.wait(), timeout=wait_seconds)
            except asyncio.TimeoutError:
                return None

    async def push_lines(self, job_id: str, lines: List[str]) -> bool:
        self._ensure_loop()
        if not lines:
            return False
        job = await self._get_job(job_id)
        if not job or job.done:
            return False
        for line in lines:
            await job.queue.put(str(line))
        return True

    async def mark_done(self, job_id: str, error: Optional[str] = None) -> bool:
        self._ensure_loop()
        job = await self._get_job(job_id)
        if not job or job.done:
            return False
        job.done = True
        job.error = error
        await job.queue.put(None)
        return True

    async def iter_lines(self, job_id: str) -> AsyncIterator[str]:
        self._ensure_loop()
        job = await self._get_job(job_id)
        if not job:
            return
        while True:
            item = await job.queue.get()
            if item is None:
                break
            yield str(item)
        await self._cleanup(job_id)

    async def get_job(self, job_id: str) -> Optional[ProxyJob]:
        self._ensure_loop()
        return await self._get_job(job_id)

    async def _get_job(self, job_id: str) -> Optional[ProxyJob]:
        self._ensure_loop()
        async with self._lock:
            self._prune_locked()
            return self._jobs.get(job_id)

    def _prune_locked(self) -> None:
        now = time.time()
        expired = [job_id for job_id, job in self._jobs.items() if job.expires_at <= now]
        for job_id in expired:
            job = self._jobs.pop(job_id, None)
            if not job:
                continue
            try:
                while not job.queue.empty():
                    job.queue.get_nowait()
            except Exception:
                pass

    async def _cleanup(self, job_id: str) -> None:
        async with self._lock:
            self._jobs.pop(job_id, None)
            try:
                self._pending.remove(job_id)
            except ValueError:
                pass
