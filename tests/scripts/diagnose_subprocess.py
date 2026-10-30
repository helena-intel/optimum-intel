import argparse
import os
import shlex
import shutil
import signal
import subprocess
import sys


SYNCHRONOUS_PARQUET_SETUP = """
import pyarrow.dataset as arrow_dataset
import pyarrow.parquet as arrow_parquet

original_parquet_format = arrow_dataset.ParquetFileFormat
scan_options = arrow_dataset.ParquetFragmentScanOptions(pre_buffer=False, use_buffered_stream=False)
synchronous_scans = 0


class SynchronousFragment:
    def __init__(self, fragment, source):
        self.fragment = fragment
        self.source = source

    def __getattr__(self, name):
        return getattr(self.fragment, name)

    def to_batches(self, **kwargs):
        global synchronous_scans
        if kwargs.get("filter") is not None:
            raise ValueError("The synchronous probe supports only unfiltered Parquet reads")
        synchronous_scans += 1
        print(
            f"Synchronous Parquet scan {synchronous_scans}: ParquetFile.iter_batches, "
            "use_threads=False, pre_buffer=False; fragment scanner bypassed",
            flush=True,
        )
        reader = arrow_parquet.ParquetFile(self.source, pre_buffer=False)
        return reader.iter_batches(
            batch_size=kwargs.get("batch_size", 131072),
            columns=kwargs.get("columns"),
            use_threads=False,
        )


class SynchronousFormat:
    def __init__(self, *args, **kwargs):
        kwargs["default_fragment_scan_options"] = scan_options
        self.format = original_parquet_format(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self.format, name)

    def make_fragment(self, file, *args, **kwargs):
        return SynchronousFragment(self.format.make_fragment(file, *args, **kwargs), file)


arrow_dataset.ParquetFileFormat = SynchronousFormat
print("Enabled probe-only synchronous Parquet control", flush=True)
"""


TEXTVQA_PROBE = """
import gc
import sys

from datasets import load_dataset

print("Loading streaming TextVQA validation", flush=True)
dataset = load_dataset("lmms-lab/textvqa", split="validation", streaming=True)
dataset = dataset.shuffle(seed=42).take(1)
records = iter(dataset)
try:
    record = next(records)
    image = record["image"]
    image.load()
    print(f"Decoded one image: {image.size}; question present: {'question' in record}", flush=True)
    if "--synchronous-parquet" in sys.argv[1:]:
        if synchronous_scans == 0:
            raise RuntimeError("Synchronous Parquet control was not exercised")
        print(f"Verified synchronous scan override was exercised {synchronous_scans} times", flush=True)
finally:
    print("Closing streaming iterator before interpreter shutdown", flush=True)
    records.close()
    print("Streaming iterator closed", flush=True)
del records, record, image, dataset
print("Collecting garbage before interpreter shutdown", flush=True)
collected = gc.collect()
print(f"Explicit cleanup completed; collected {collected} objects", flush=True)
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
    parser.add_argument("--synchronous-parquet", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    if args.synchronous_parquet and not args.probe_textvqa:
        parser.error("--synchronous-parquet requires --probe-textvqa")
    command = args.command
    if command[:1] == ["--"]:
        command = command[1:]
    if args.probe_textvqa:
        if command:
            parser.error("--probe-textvqa cannot be combined with a command")
        probe_code = TEXTVQA_PROBE
        if args.synchronous_parquet:
            probe_code = SYNCHRONOUS_PARQUET_SETUP + probe_code
        command = [sys.executable, "-u", "-c", probe_code]
        if args.synchronous_parquet:
            command.append("--synchronous-parquet")
    if not command:
        parser.error("provide a command or --probe-textvqa")
    return run_with_diagnostics(command, args.timeout)


if __name__ == "__main__":
    sys.exit(main())
