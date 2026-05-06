from nexus.core.engine import TaskEngine
from nexus.core.context import SessionContext

e = TaskEngine()
print(f"Engine: {e}")
print(f"Context: {e.context}")
if e.context:
    print(f"History: {e.context.conversation_history}")
else:
    print("Context is NONE!")
