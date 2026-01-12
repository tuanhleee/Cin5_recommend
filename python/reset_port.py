import os
import subprocess

def kill_port(port: int):
    """
    Kill the process that is listening on the given port.
    Equivalent to: lsof -t -i:{port} | xargs kill -9
    """
    try:
        # Find PID(s)
        result = subprocess.check_output(
            ["lsof", "-t", f"-i:{port}"],
            stderr=subprocess.STDOUT
        ).decode().strip()

        if not result:
            print(f"No process is using port {port}.")
            return

        pids = result.split("\n")
        for pid in pids:
            print(f"Killing PID {pid} on port {port}...")
            os.system(f"kill -9 {pid}")

        print(f"Port {port} is now free.")

    except subprocess.CalledProcessError:
        print(f"No process is using port {port}.")
kill_port(8000)