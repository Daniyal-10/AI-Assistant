from nexus.core.engine import TaskEngine
from nexus.ai.router import IntentRouter
from nexus.core.context import SessionContext
import sys

try:
    engine = TaskEngine(SessionContext())
    router = IntentRouter()

    inputs = [
        "Hello, how are you?",
        "Build a Python script that prints numbers 1 to 10",
        "Explain this code: def add(a,b): return a+b",
        "list files in workspace"
    ]

    for i in inputs:
        print(f"\n--- Testing: {i} ---")
        ir = router.route(i)
        print(f"Intent: {ir.intent.value}")
        task = engine.run(i, ir)
        print(f"Result Status: {task.status.name}")
        if task.result:
            print(f"Summary: {task.result.summary[:200]}...")
except Exception as e:
    print(f"ERROR: {e}")
    sys.exit(1)
