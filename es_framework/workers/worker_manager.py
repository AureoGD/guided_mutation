import multiprocessing as mp
import numpy as np
from guided_mutation.es_framework.workers.worker import worker_loop
from queue import Empty
import time
import pickle
import os


class WorkerManager:

    def __init__(self, num_workers, config, task_timeout=10):
        self.ctx = mp.get_context("spawn")
        print(f"[WorkerManager] Start method: {self.ctx.get_start_method()}")

        self.num_workers = num_workers
        self.config = config
        self.task_timeout = task_timeout

        self.task_queue = self.ctx.Queue()
        self.result_queue = self.ctx.Queue()
        self.start_event = self.ctx.Event()

        self.copy_task_queue = {}
        self.pending_tasks = {}
        self.task_order = []

        self.heartbeats = []
        self.worker_current_task = self.ctx.Array('i', [-1] * num_workers)

        self.deferred_kills = set()
        self.workers = []

        for worker_id in range(num_workers):
            hb = self.ctx.Value('d', time.time())
            self.heartbeats.append(hb)
            self._spawn_worker(worker_id)

        print(f"[WorkerManager] {num_workers} workers criados.")
        self.start_event.set()

        # pasta para salvar falhas
        os.makedirs("failed_tasks", exist_ok=True)

    # ----------------------------------------
    def _spawn_worker(self, worker_id):
        hb = self.heartbeats[worker_id]

        with hb.get_lock():
            hb.value = time.time()

        p = self.ctx.Process(
            target=worker_loop,
            args=(
                worker_id,
                self.task_queue,
                self.result_queue,
                self.config,
                self.start_event,
                hb,
                self.worker_current_task,
            ),
        )

        p.start()

        if worker_id < len(self.workers):
            self.workers[worker_id] = p
        else:
            self.workers.append(p)

    # ----------------------------------------
    def submit(self, tasks):
        self.copy_task_queue.clear()

        for task in tasks:
            t_id, sp_id, ind_id = task[0], task[1], task[2]

            self.pending_tasks[t_id] = {"species_id": sp_id, "ind_id": ind_id}
            self.task_order.append(t_id)
            self.copy_task_queue[t_id] = task
            self.task_queue.put_nowait(task)

    # ----------------------------------------
    def collect(self, n_tasks, timeout=None):

        if timeout is None:
            timeout = self.task_timeout

        collected = 0
        start_time = time.time()

        while collected < n_tasks:

            try:
                result = self.result_queue.get(timeout=0.1)
                t_id = result.get("task_id")

                self.pending_tasks.pop(t_id, None)
                if t_id in self.task_order:
                    self.task_order.remove(t_id)

                yield result
                collected += 1
                start_time = time.time()
                continue

            except Empty:
                pass

            # ----------------------------------------
            # TIMEOUT / CRASH DETECTION
            # ----------------------------------------
            elapsed = time.time() - start_time

            if elapsed > timeout and self.task_order:
                stuck_worker_id = self._find_stuck_worker(timeout)

                if stuck_worker_id is not None and stuck_worker_id not in self.deferred_kills:

                    stuck_task_id = self.worker_current_task[stuck_worker_id]

                    if stuck_task_id != -1 and stuck_task_id in self.pending_tasks:
                        failed_task_id = stuck_task_id
                    else:
                        failed_task_id = self.task_order[0]

                    task_meta = self.pending_tasks.pop(failed_task_id, {"species_id": -1, "ind_id": -1})
                    if failed_task_id in self.task_order:
                        self.task_order.remove(failed_task_id)

                    self.deferred_kills.add(stuck_worker_id)

                    print(
                        f"[WorkerManager] Worker {stuck_worker_id} marcado para kill deferido (Task {failed_task_id})")

                    # 🔥 SALVAR TASK (SIMPLES)
                    task_data = self.copy_task_queue.get(failed_task_id)

                    if task_data is not None:
                        filename = f"failed_tasks/failed_task_{failed_task_id}.pkl"
                        with open(filename, "wb") as f:
                            pickle.dump(task_data, f)
                        print(f"[DEBUG] Task {failed_task_id} salva em {filename}")
                    else:
                        print(f"[WARNING] Task {failed_task_id} não encontrada para salvar")

                    yield self._generate_failure_dict(
                        task_id=failed_task_id,
                        species_id=task_meta["species_id"],
                        ind_id=task_meta["ind_id"],
                    )

                    collected += 1
                    start_time = time.time()

            time.sleep(0.01)

        # ----------------------------------------
        # CLEANUP
        # ----------------------------------------
        if self.deferred_kills:
            print(f"[WorkerManager] Aplicando kills deferidos: {self.deferred_kills}")
            for worker_id in list(self.deferred_kills):
                self._replace_stuck_worker(worker_id)
            self.deferred_kills.clear()

    # ----------------------------------------
    def _find_stuck_worker(self, timeout):
        now = time.time()

        for i, (p, hb) in enumerate(zip(self.workers, self.heartbeats)):

            if not p.is_alive():
                print(f"[WorkerManager] Worker {i} (PID {p.pid}) morreu. exitcode={p.exitcode}")
                return i

            with hb.get_lock():
                last_beat = hb.value

            if now - last_beat > timeout:
                print(f"[WorkerManager] Worker {i} sem heartbeat ({now - last_beat:.2f}s)")
                return i

        return None

    # ----------------------------------------
    def _replace_stuck_worker(self, worker_id):
        p = self.workers[worker_id]

        print(f"[WorkerManager] Encerrando Worker {worker_id} PID={p.pid} exitcode={p.exitcode}")

        if p.is_alive():
            p.kill()

        p.join(timeout=3)

        self.worker_current_task[worker_id] = -1

        self._spawn_worker(worker_id)

        print(f"[WorkerManager] Worker {worker_id} reiniciado PID={self.workers[worker_id].pid}")

    # ----------------------------------------
    def _generate_failure_dict(self, task_id, species_id, ind_id):
        return {
            "status": "timeout_error",
            "task_id": task_id,
            "species_id": species_id,
            "ind_id": ind_id,
            "reward": np.float32(-1e4),
            "success_flag": 0,
            "trajectory": [],
            "train_metrics": {},
            "refined_params": {},
        }

    # ----------------------------------------
    def shutdown(self):

        for _ in range(self.num_workers):
            self.task_queue.put(None)

        for p in self.workers:
            p.join(timeout=5)
            if p.is_alive():
                print(f"[WorkerManager] Forçando kill PID={p.pid}")
                p.kill()
                p.join()

        print("[WorkerManager] Shutdown completo.")
