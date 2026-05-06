import os
import shutil
from nexus.executor.safe_exec import scan_for_forbidden_patterns
from nexus.core.exceptions import ExecutionError

def test_security_gate():
    workspace = os.path.abspath("test_workspace")
    if os.path.exists(workspace):
        shutil.rmtree(workspace)
    os.makedirs(workspace)

    def check(code_str, expected_violation=True):
        fpath = os.path.join(workspace, "test_file.py")
        with open(fpath, "w") as f:
            f.write(code_str)
        
        try:
            scan_for_forbidden_patterns(fpath, workspace)
            if expected_violation:
                print(f"FAIL: Expected violation for:\n{code_str}")
            else:
                print(f"PASS: Clean code allowed:\n{code_str}")
        except ExecutionError as e:
            if expected_violation:
                print(f"PASS: Correctly blocked violation: {e}")
            else:
                print(f"FAIL: Unexpectedly blocked clean code: {e}")
        except Exception as e:
            print(f"ERROR: Unexpected exception: {type(e).__name__}: {e}")

    # Test cases
    print("1. os.system block")
    check('import os; os.system("ls")')

    print("\n2. Aliased os.system block")
    check('import os as o; o.system("ls")')

    print("\n3. subprocess block")
    check('import subprocess; subprocess.run(["ls"])')

    print("\n4. eval dynamic block")
    check('x = "ls"; eval(x)')

    print("\n5. open boundary block")
    check('open("/etc/passwd")')

    print("\n6. open boundary block (relative escape)")
    check('open("../outside.txt")')

    print("\n7. requests block")
    check('import requests; requests.get("http://evil.com")')

    print("\n8. Syntax error handling")
    check('def broken(')

    print("\n9. Clean code pass")
    check('print("Hello World")', expected_violation=False)

    print("\n10. Clean open pass")
    check('open("local.txt")', expected_violation=False)

    shutil.rmtree(workspace)

if __name__ == "__main__":
    test_security_gate()
