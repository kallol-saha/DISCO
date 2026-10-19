"""
Simulation budget: a file-locked counter shared by all search processes.
"""
import fcntl
import json
import os


class SimBudget:
    """Simulator-call budget shared across processes through a locked JSON file."""

    def __init__(self, path: str, total: int = None):
        """Open (or create) the budget file and set its total if given.

        Args:
            path: Budget JSON path.
            total: Budget β in simulator calls; None keeps the stored total.
        """
        self.path = path
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with self._locked() as state:
            if total is not None:
                state["total"] = int(total)

    def _locked(self):
        """Return a context manager that locks, loads and saves the budget state."""
        return _LockedState(self.path)

    def charge(self, env_index: int, calls: int) -> int:
        """Add simulator calls to one env's count.

        Args:
            env_index: Env the calls were spent on.
            calls: Number of simulator calls.

        Returns:
            New total used over all envs.
        """
        with self._locked() as state:
            key = str(env_index)
            state["used"][key] = state["used"].get(key, 0) + int(calls)
            return sum(state["used"].values())

    def used(self, env_index: int = None) -> int:
        """Return the simulator calls used by one env, or by all envs.

        Args:
            env_index: Env to query, or None for the total over all envs.
        """
        with self._locked() as state:
            if env_index is None:
                return sum(state["used"].values())
            return state["used"].get(str(env_index), 0)

    @property
    def total(self):
        """Return the budget β, or None if it is unset."""
        with self._locked() as state:
            return state.get("total")

    def exhausted(self) -> bool:
        """Return True once the calls used over all envs reach the budget."""
        with self._locked() as state:
            total = state.get("total")
            return total is not None and sum(state["used"].values()) >= total


class _LockedState:
    """Read-modify-write of the budget JSON under an exclusive file lock."""

    def __init__(self, path):
        """Store the budget JSON path.

        Args:
            path: Budget JSON path; the lock file is `<path>.lock`.
        """
        self.path = path

    def __enter__(self):
        """Take the exclusive lock and return the loaded state dict."""
        self._fh = open(self.path + ".lock", "a")
        fcntl.flock(self._fh, fcntl.LOCK_EX)
        if os.path.exists(self.path):
            with open(self.path) as f:
                self.state = json.load(f)
        else:
            self.state = {"total": None, "used": {}}
        return self.state

    def __exit__(self, *exc):
        """Write the state back atomically and release the lock.

        Args:
            *exc: Exception info from the with-block (never suppressed).
        """
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.state, f, indent=1)
        os.replace(tmp, self.path)
        fcntl.flock(self._fh, fcntl.LOCK_UN)
        self._fh.close()
        return False
