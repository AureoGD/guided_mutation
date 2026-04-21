import multiprocessing as mp
from es_framework.workers.worker import worker_loop


# -------------------------------------------------
# WORKER MANAGER
# -------------------------------------------------
class WorkerManager:

    def __init__(self, num_workers, config):

        self.num_workers = num_workers
        self.config = config

        self.task_queue = mp.Queue()
        self.result_queue = mp.Queue()

        # ----------------------------------------
        # SYNC EVENT (BARRIER)
        # ----------------------------------------
        self.start_event = mp.Event()

        self.workers = []

        # ----------------------------------------
        # CREATE WORKERS
        # ----------------------------------------
        for worker_id in range(num_workers):

            p = mp.Process(target=worker_loop,
                           args=(worker_id, self.task_queue, self.result_queue, self.config, self.start_event))

            p.start()
            self.workers.append(p)

        print("[WorkerManager] All workers created. Releasing start signal...")

        # ----------------------------------------
        # RELEASE ALL WORKERS
        # ----------------------------------------
        self.start_event.set()

    # ----------------------------------------
    # SUBMIT TASKS
    # ----------------------------------------
    def submit(self, tasks):

        for task in tasks:
            self.task_queue.put(task)

    # ----------------------------------------
    # COLLECT RESULTS (GENERATOR)
    # ----------------------------------------
    def collect(self, n_tasks):

        for _ in range(n_tasks):
            result = self.result_queue.get()
            yield result

    # ----------------------------------------
    # SHUTDOWN
    # ----------------------------------------
    def shutdown(self):

        for _ in range(self.num_workers):
            self.task_queue.put(None)

        for p in self.workers:
            p.join()

        print("[WorkerManager] All workers stopped.")
