import argparse
import os
import shlex
import shutil
import signal
import subprocess
import sys


TEXTVQA_PROBE = """
from datasets import load_dataset

print("Loading streaming TextVQA validation", flush=True)
dataset = load_dataset("lmms-lab/textvqa", split="validation", streaming=True)
dataset = dataset.shuffle(seed=42).take(1)
records = iter(dataset)
record = next(records)
image = record["image"]
image.load()
print(f"Decoded one image: {image.size}; question present: {'question' in record}", flush=True)
print("Dataset-only probe entering interpreter shutdown", flush=True)
"""


def dump_native_stacks(process_id):
    if not sys.platform.startswith("linux") or shutil.which("gdb") is None:
        print("Native stacks unavailable: Linux and gdb are required", flush=True)
        return
    command = [
        "gdb",
        "--batch",
        "--nx",
        "-p",
        str(process_id),
        "-ex",
        "set pagination off",
        "-ex",
        "set print frame-arguments none",
        "-ex",
        "info threads",
        "-ex",
        "thread apply all bt",
        "-ex",
        "detach",
    ]
    if os.geteuid() != 0 and shutil.which("sudo") is not None:
        command = ["sudo", "-n", *command]
    try:
        subprocess.run(command, timeout=30, check=False)
    except (OSError, subprocess.TimeoutExpired) as error:
        print(f"Native stack collection failed: {error}", flush=True)


def run_with_diagnostics(command, timeout):
    print(f"Starting child (timeout {timeout}s): {shlex.join(command)}", flush=True)
    with subprocess.Popen(command, start_new_session=os.name == "posix") as process:
        print(f"Child PID: {process.pid}", flush=True)
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            print(f"Child exceeded {timeout}s; collecting native stacks before killing it", flush=True)
            try:
                dump_native_stacks(process.pid)
            finally:
                if os.name == "posix":
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                elif process.poll() is None:
                    process.kill()
                process.wait(timeout=15)
            return 124
    print(f"Child exited with return code {process.returncode}", flush=True)
    return process.returncode if process.returncode >= 0 else 128 - process.returncode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--probe-textvqa", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    command = args.command
    if command[:1] == ["--"]:
        command = command[1:]
    if args.probe_textvqa:
        if command:
            parser.error("--probe-textvqa cannot be combined with a command")
        command = [sys.executable, "-u", "-c", TEXTVQA_PROBE]
    if not command:
        parser.error("provide a command or --probe-textvqa")
    return run_with_diagnostics(command, args.timeout)


if __name__ == "__main__":
    sys.exit(main())
