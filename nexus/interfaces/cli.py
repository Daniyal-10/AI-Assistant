"""
NEXUS CLI — Primary interface.
"""
import sys
from nexus.core.engine import TaskEngine
from nexus.core.context import SessionContext
from nexus.core.task import Task, TaskStatus
from nexus.ai.router import IntentRouter, IntentType
from nexus.interfaces.formatter import ResponseFormatter
from nexus.utils.logger import get_logger

logger = get_logger(__name__)

BANNER = """
╔═══════════════════════════════════════╗
║          NEXUS v0.1.0 — LOCAL         ║
║   Generate → Execute → Fix → Return   ║
╚═══════════════════════════════════════╝
"""


def print_result(task: Task) -> None:
    print("\n" + "─" * 50)

    if task.status == TaskStatus.DONE and task.result and task.result.success:
        print("✅ TASK COMPLETE")
        print(f"   {task.result.summary}")
        print(f"   Fix iterations: {task.result.iterations_used}")

        if task.result.output_path:
            print(f"   Output: {task.result.output_path}")

    else:
        print("❌ TASK FAILED")

        reason = task.result.summary if task.result else "Unknown error"
        print(f"   Reason: {reason}")

        if task.workspace_path:
            print(f"   Workspace (debug): {task.workspace_path}")

    print("─" * 50 + "\n")


def run_interactive() -> None:
    formatter = ResponseFormatter()
    formatter.print_welcome()

    context = SessionContext()
    engine = TaskEngine(context=context)
    router = IntentRouter()

    while True:
        try:
            user_input = input("nexus> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n")
            formatter.format_chat("Disconnecting. Have a productive day.")
            sys.exit(0)

        if not user_input:
            continue

        if user_input.lower() in {"exit", "quit", "q"}:
            context.save()
            formatter.format_chat("Session saved. Goodbye.")
            sys.exit(0)

        # 1. Routing
        formatter.step("Analyzing request...")
        intent_result = router.route(user_input)
        formatter.print_intent_identified(
            intent_result.intent.value, 
            intent_result.confidence
        )

        # 2. Execution
        try:
            formatter.step("Processing...")
            result_task = engine.run(user_input, intent_result)
            
            # 3. Formatted Response
            if intent_result.intent == IntentType.CHAT:
                formatter.format_chat(result_task.result.summary)
            elif intent_result.intent == IntentType.CODE:
                formatter.format_code_intel(result_task.result.summary)
            elif intent_result.intent == IntentType.TASK:
                formatter.format_task_result(result_task)
            else:
                if result_task.result:
                    print(f"\n{result_task.result.summary}\n")

        except KeyboardInterrupt:
            print("\n")
            formatter.format_chat("Action cancelled by user.")

        except Exception as e:
            logger.exception("CLI error")
            formatter.print_error(str(e), "Check the logs for more details.")


def run_single(task_input: str) -> int:
    if not task_input.strip():
        print("Error: empty task input.")
        return 1

    context = SessionContext()
    engine = TaskEngine(context=context)
    task = Task(raw_input=task_input)
    context.add_message("user", task_input)

    try:
        result_task = engine.run(task)
        print_result(result_task)
        context.save()

        return 0 if (result_task.result and result_task.result.success) else 1

    except Exception as e:
        logger.exception("CLI single-run error")
        print(f"\n❌ Unexpected error: {e}")
        return 1
