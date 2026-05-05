from nexus.core.engine import TaskEngine
from nexus.ai.router import IntentRouter
from nexus.core.context import SessionContext
import sys

try:
    engine = TaskEngine(SessionContext())
    router = IntentRouter()

    user_input = "Explain this code: def add(a,b): return a+b"
    print(f"\n--- Testing: {user_input} ---")
    ir = router.route(user_input)
    print(f"Intent: {ir.intent.value}")
    task = engine.run(user_input, ir)
    print(f"Result Status: {task.status.name}")
    if task.result:
        print(f"Summary: {task.result.summary}")
except Exception as e:
    print(f"ERROR: {e}")
    sys.exit(1)
