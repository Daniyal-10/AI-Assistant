import os
import sys

# Ensure the module can be imported
sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/.."))

from nexus.utils.history import TaskHistory, TaskRecord
from nexus.core.restorer import SessionRestorer
from nexus.core.context import SessionContext
import json
import logging

logging.basicConfig(level=logging.INFO)

# Create a dummy history store and inject some records
class DummyTaskHistory(TaskHistory):
    def _get_current_file(self):
        import pathlib
        return pathlib.Path("dummy_history.jsonl")

store = DummyTaskHistory()
# Write dummy records
with open("dummy_history.jsonl", "w") as f:
    r1 = TaskRecord(intent="TASK", plan_summary="Fix the bug", execution_status="PASS")
    r2 = TaskRecord(intent="CODE", plan_summary="Refactor", execution_status="FAIL")
    f.write(json.dumps(r1.__dict__) + "\n")
    f.write(json.dumps(r2.__dict__) + "\n")

print("--- Testing Restorer ---")
payload = SessionRestorer.restore(store)
print(f"Loaded payload: {payload}")

print("\n--- Testing Context Injection ---")
context = SessionContext(warm_history=payload)
print("Context Output:")
print(context.get_recent_context())

# Cleanup
import os
os.remove("dummy_history.jsonl")
