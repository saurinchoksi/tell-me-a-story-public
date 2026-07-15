"""Run a heavy model in a fresh, short-lived subprocess — the finish-and-free pattern.

A model on Apple Silicon (Whisper, pyannote, Qwen via mlx-lm, Gemma via mlx-vlm) holds
the Mac's Metal/MPS memory in a caching allocator that is NOT reliably released when the
model object is deleted — and one framework's held memory (pyannote/torch) can block
another (mlx) from loading. The only sure way to free it is to END the process: on exit
the OS reclaims all of it. So each heavy step runs its work in a spawned subprocess that
loads the model, does the work, returns the result, and exits.

One venv now: the subprocess is the same interpreter (spawn), so every model stack is
available. The worker function must be module-level (picklable by reference); its args
and return value must be picklable. This is the single primitive behind the word-fixer
(normalize.py), the M9b name judge, and the story-name auditor.
"""
import multiprocessing
import queue as _queue


def _worker_wrapper(result_queue, fn, args, kwargs):
    """Run fn in the child and ship (ok, value-or-exception) back through the queue."""
    try:
        result_queue.put((True, fn(*args, **kwargs)))
    except Exception as e:  # propagate the worker's failure to the parent
        try:
            result_queue.put((False, e))
        except Exception:  # exception itself isn't picklable — send a stand-in
            result_queue.put((False, RuntimeError(repr(e))))


def run_model(fn, /, *args, timeout: int = 600, **kwargs):
    """Run fn(*args, **kwargs) in a fresh spawned subprocess and return its result.

    The subprocess is KILLED if it doesn't finish within `timeout` seconds — bounding
    wall-clock time and freeing its GPU memory at the cap — and exits normally otherwise,
    so the next model loads into a clean GPU slate. fn must be a module-level callable;
    its args/kwargs and return value must be picklable. Raises TimeoutError on overrun;
    any exception fn raised is re-raised in the caller.
    """
    ctx = multiprocessing.get_context("spawn")
    result_queue = ctx.Queue()
    proc = ctx.Process(target=_worker_wrapper, args=(result_queue, fn, args, kwargs))
    proc.start()
    try:
        # get() (not join()) drains the result pipe, so a large return can't deadlock,
        # and it bounds the wait: a hung worker raises Empty here and is killed below.
        ok, payload = result_queue.get(timeout=timeout)
    except _queue.Empty:
        proc.kill()
        raise TimeoutError(f"model subprocess timed out after {timeout}s")
    finally:
        proc.join()  # reap the child (it has exited normally or been killed)
    if not ok:
        raise payload
    return payload
