from nexus.ai.router import IntentRouter
import sys

try:
    router = IntentRouter()
    print("Testing 'Hello, how are you?'")
    res1 = router.route("Hello, how are you?")
    print(f"Result 1 Intent: {res1.intent.value}")

    print("\nTesting 'Build a Python script that prints numbers 1 to 10'")
    res2 = router.route("Build a Python script that prints numbers 1 to 10")
    print(f"Result 2 Intent: {res2.intent.value}")
except Exception as e:
    print(f"ERROR: {e}")
    sys.exit(1)
