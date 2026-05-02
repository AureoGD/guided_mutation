import time
import numpy as np
from queue import Empty
import multiprocessing as mp
from guided_mutation.es_framework.workers.worker import worker_loop

FLAG_SIMULATE = 0
FLAG_REFINE = 1


class WorkerManager:

    def __init__(self, num_workers, config, task_timeout=10):
        self.num_workers = num_workers
        self.config = config
        self.task_timeout = task_timeout

        self.worker_death_count = 0

        self.ctx = mp.get_context("spawn")

        # Task and result queues
        self.task_queue = self.ctx.Queue()
        self.result_queue = self.ctx.Queue()

        # Workers and heartbeats
        self.workers = []
        self.heartbeats = []
        self.start_event = self.ctx.Event()
        self.pending_tasks = {}

        # Spawn workers and release them
        for worker_id in range(num_workers):
            self._spawn_worker(worker_id)

        self.start_event.set()

        print("[WorkerManager] All workers created and ready for tasks.")

    def _spawn_worker(self, worker_id):
        # Heartbeat array shared with the worker: [timestamp, sim_step, env_id, task_id]
        # Used by the manager to detect crashes and identify the stuck task
        hb = self.ctx.Array('d', [time.time(), -1, -1, -1])

        if worker_id < len(self.heartbeats):
            self.heartbeats[worker_id] = hb  # replace on restart
        else:
            self.heartbeats.append(hb)  # first spawn

        # Create and start the worker proces
        p = self.ctx.Process(target=worker_loop,
                             args=(worker_id, self.task_queue, self.result_queue, self.config, hb, self.start_event))
        p.start()

        if worker_id < len(self.workers):
            self.workers[worker_id] = p  # replace on restart
        else:
            self.workers.append(p)  # first spawn

    def submit(self, tasks):
        self.pending_tasks.clear()  # assumes previous batch fully collected
        for task in tasks:
            t_id = task["task_id"]
            self.pending_tasks[t_id] = task
            self.task_queue.put_nowait(task)

    def collect(self, n_tasks):
        collected = 0
        while collected < n_tasks:
            try:
                result = self.result_queue.get(timeout=0.1)
                self.pending_tasks.pop(result["task_id"])
                yield result
                collected += 1
                continue
            except Empty:
                pass

            # Check for stuck/dead workers
            stuck_id = self._find_stuck_worker()
            if stuck_id is not None:
                task_id = int(self.heartbeats[stuck_id][3])
                if task_id == -1:
                    self._replace_stuck_worker(stuck_id)
                else:
                    task = self.pending_tasks.pop(task_id)
                    self._replace_stuck_worker(stuck_id)
                    yield self._failure_result(task, stuck_id)
                    collected += 1

    def _find_stuck_worker(self):
        now = time.time()

        for i, (p, hb) in enumerate(zip(self.workers, self.heartbeats)):

            # Condition 1 - process is dead
            if not p.is_alive():
                self.worker_death_count += 1
                return i

            # Condition 2 - stuck (heartbeat too old AND not idle)
            if now - hb[0] > self.task_timeout and hb[3] != -1:
                self.worker_death_count += 1
                return i

        return None

    def _replace_stuck_worker(self, worker_id):
        p = self.workers[worker_id]

        if p.is_alive():
            p.kill()  # force kill if still running (segfault, stuck, etc.)

        p.join(timeout=3)

        self._spawn_worker(worker_id)

    def _failure_result(self, task, stuck_id):
        return {
            "type": "simulation_result" if task["flag"] == FLAG_SIMULATE else "refined_result",
            "task_id": task["task_id"],
            "species_id": task["species_id"],
            "ind_id": task["ind_id"],
            "reward": np.float32(-1e4),
            "success_percent": 0.0,
            "status": "error",
            "extra": {
                "error": "timeout_error",
                "worker_id": stuck_id,
            }
        }

    def shutdown(self):
        for _ in range(self.num_workers):
            self.task_queue.put(None)

        for p in self.workers:
            p.join(timeout=5)
            if p.is_alive():
                p.kill()
                p.join()

        print("[WorkerManager] Complete Shutdown.")
