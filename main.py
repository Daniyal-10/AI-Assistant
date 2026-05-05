#!/usr/bin/env python3
"""
NEXUS Entry Point

Usage:
    python main.py                          # Interactive mode
    python main.py "Build a FastAPI app"    # Single task mode
    python main.py --help
"""
import sys
from nexus.interfaces.cli import run_interactive, run_single
from nexus.utils.logger import get_logger

logger = get_logger("nexus.main")


def main() -> None:
    try:
        args = sys.argv[1:]

        logger.info("NEXUS starting...")

        if "--help" in args or "-h" in args:
            print(__doc__)
            sys.exit(0)

        if args:
            # Single task mode
            task_input = " ".join(args)
            exit_code = run_single(task_input)
            sys.exit(exit_code)
        else:
            # Interactive mode
            run_interactive()

    except KeyboardInterrupt:
        print("\nShutting down NEXUS.")
        sys.exit(0)

    except Exception as e:
        logger.exception("Fatal error in main")
        print(f"\n❌ Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
