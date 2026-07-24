"""
Generate a markdown report summarizing the latest workflow run for test_openvino_genai_aipc.yml.

Fetches job data and logs from the GitHub Actions API, parses pytest results per model,
and produces a markdown table with models in rows, MTL/LNL columns with CPU/GPU/NPU subcolumns.

Requirements:
    pip install requests

Usage:
    # Set GITHUB_TOKEN environment variable (needs repo/actions read access)
    # Or have `gh` CLI authenticated (token will be auto-detected)
    set GITHUB_TOKEN=ghp_...
    python generate_genai_report.py

    # Optionally specify a run ID:
    python generate_genai_report.py --run-id 29916107736

    # Save to file (markdown):
    python generate_genai_report.py --output report.md

    # Generate HTML report:
    python generate_genai_report.py --html --output report.html
"""

import argparse
import html as html_lib
import os
import re
import subprocess
import sys
from collections import defaultdict

import requests

OWNER = os.environ.get("GITHUB_REPOSITORY_OWNER", "helena-intel")
REPO_NAME = os.environ.get("GITHUB_REPOSITORY", "helena-intel/optimum-intel").split("/")[-1]
WORKFLOW_FILE = "test_openvino_genai_aipc.yml"
API_BASE = f"https://api.github.com/repos/{OWNER}/{REPO_NAME}"


def get_headers():
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        # Try to get token from gh CLI
        try:
            result = subprocess.run(["gh", "auth", "token"], capture_output=True, text=True, check=True)
            token = result.stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass
    if not token:
        print("ERROR: GITHUB_TOKEN environment variable is required (or authenticate via `gh auth login`).",
              file=sys.stderr)
        sys.exit(1)
    return {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json",
    }


def get_latest_run(run_id=None):
    """Get the latest completed workflow run, or a specific run by ID."""
    headers = get_headers()
    if run_id:
        url = f"{API_BASE}/actions/runs/{run_id}"
        resp = requests.get(url, headers=headers)
        resp.raise_for_status()
        return resp.json()
    else:
        url = f"{API_BASE}/actions/workflows/{WORKFLOW_FILE}/runs"
        params = {"per_page": 10, "status": "completed"}
        resp = requests.get(url, headers=headers, params=params)
        resp.raise_for_status()
        runs = resp.json()["workflow_runs"]
        if not runs:
            print("No completed workflow runs found.", file=sys.stderr)
            sys.exit(1)
        return runs[0]


def get_jobs(run_id):
    """Get all jobs for a workflow run."""
    headers = get_headers()
    jobs = []
    page = 1
    while True:
        url = f"{API_BASE}/actions/runs/{run_id}/jobs"
        resp = requests.get(url, headers=headers, params={"per_page": 100, "page": page})
        resp.raise_for_status()
        data = resp.json()
        jobs.extend(data["jobs"])
        if len(jobs) >= data["total_count"]:
            break
        page += 1
    return jobs


def download_job_log(job_id):
    """Download log for a single job. Returns the log text or None on failure."""
    headers = get_headers()
    url = f"{API_BASE}/actions/jobs/{job_id}/logs"
    resp = requests.get(url, headers=headers, allow_redirects=True)
    if resp.status_code == 200:
        return resp.text
    return None


def parse_job_name(job_name):
    """Parse job name like 'test (4.57.6, GPU, MTL)' into (version, device, runner)."""
    match = re.match(r"test\s*\(([^,]+),\s*([^,]+),\s*([^)]+)\)", job_name)
    if match:
        return match.group(1).strip(), match.group(2).strip(), match.group(3).strip()
    return None, None, None


def parse_versions_from_log(log_text):
    """Extract package versions from the 'Verify imports' step output.

    Looks for lines like:
        openvino_genai 2026.3.0.0-3272-e9c0cf80d50
        openvino 2026.3.0-22446-ce03a15d415-releases/2026/3
        transformers 4.57.6
        torch 2.13.0+cpu

    Returns dict of {package_name: version_string}.
    """
    versions = {}
    timestamp_re = re.compile(r"^\d{4}-\d{2}-\d{2}T[\d:.]+Z\s?")
    for line in log_text.split("\n"):
        stripped = timestamp_re.sub("", line).strip()
        for pkg in ("openvino_genai", "openvino", "transformers", "torch"):
            if stripped.startswith(f"{pkg} "):
                versions[pkg] = stripped[len(pkg) + 1:].strip()
                break
    return versions


def parse_test_results_from_log(log_text):
    """Parse pytest output from a GitHub Actions job log to extract per-test results.

    The log format is:
        2026-07-22T11:35:00.6366112Z tests/openvino/test_genai.py::ClassName::method_name
        ... (optional log lines) ...
        2026-07-22T11:35:06.8379394Z PASSED                                   [ xx%]

    Or inline (for some tests):
        2026-07-22T... tests/openvino/test_genai.py::Class::method SKIPPED [xx%]

    Returns dict of {(class_name, test_method): status} where status is 'passed', 'failed', 'skipped', or 'error'.
    """
    results = {}
    lines = log_text.split("\n")

    # Strip timestamp prefix from lines: "2026-07-22T11:35:00.6366112Z "
    timestamp_re = re.compile(r"^\d{4}-\d{2}-\d{2}T[\d:.]+Z\s?")

    current_test = None  # (class_name, method_name)

    for line in lines:
        stripped = timestamp_re.sub("", line).strip()

        # Check for inline result: "test_genai.py::Class::method PASSED/FAILED/SKIPPED [xx%]"
        inline_match = re.match(
            r"(?:tests/openvino/)?test_genai\.py::(\w+)::(\w+)\s+(PASSED|FAILED|SKIPPED)",
            stripped,
        )
        if inline_match:
            class_name = inline_match.group(1)
            method = inline_match.group(2)
            status = inline_match.group(3).lower()
            results[(class_name, method)] = status
            current_test = None
            continue

        # Check for test name line: "tests/openvino/test_genai.py::ClassName::method_name"
        test_match = re.match(
            r"(?:tests/openvino/)?test_genai\.py::(\w+)::(\w+)\s*$",
            stripped,
        )
        if test_match:
            current_test = (test_match.group(1), test_match.group(2))
            continue

        # Check for standalone result line: "PASSED  [ xx%]" or "FAILED  [ xx%]" or "SKIPPED (reason)  [ xx%]"
        # These always end with a percentage indicator like "[ xx%]"
        result_match = re.match(r"(PASSED|FAILED|SKIPPED)\b", stripped)
        if result_match and current_test and re.search(r"\[\s*\d+%\]", stripped):
            status = result_match.group(1).lower()
            results[current_test] = status
            current_test = None
            continue

        # Check for crash/access violation (indicates current test errored)
        if current_test and "fatal exception" in stripped.lower():
            results[current_test] = "error"
            current_test = None
            continue

    return results


def extract_model_from_test_name(test_method):
    """Extract model architecture name from test method like 'test_compare_outputs_00_gpt2'.

    parameterized adds a numeric prefix: test_compare_outputs_00_gpt_bigcode, test_compare_outputs_01_bloom, etc.
    Returns None if the test method doesn't match the expected pattern.
    """
    match = re.match(r"test_compare_outputs_(?:vlm_)?\d+_(.+)", test_method)
    if match:
        return match.group(1)
    return None


def extract_test_category(class_name):
    """Map test class name to a category."""
    mapping = {
        "LLMPipelineTestCase": "LLM",
        "VLMPipelineTestCase": "VLM",
        "Speech2TextPipelineTestCase": "Speech2Text",
        "Text2SpeechPipelineTestCase": "Text2Speech",
        "LLMPipelineWithEagle3TestCase": "Eagle3",
        "VLMPipelineWithEagle3TestCase": "Eagle3-VLM",
    }
    return mapping.get(class_name, class_name)


def build_report(run_info, jobs, job_logs):
    """Build the markdown report from jobs and their logs.

    Args:
        run_info: Workflow run metadata from API
        jobs: List of job objects from API
        job_logs: Dict of {job_id: log_text}
    """
    # Structure: results[version][(category, model_name)] = {(runner, device): status}
    results = defaultdict(lambda: defaultdict(dict))
    job_conclusions = {}
    # Track versions per transformers version (from any job with that version)
    version_info = {}  # {transformers_version: {pkg: version_str}}

    for job in jobs:
        job_name = job["name"]
        version, device, runner = parse_job_name(job_name)
        if not version:
            continue

        conclusion = job.get("conclusion", "unknown")
        job_conclusions[(version, device, runner)] = conclusion

        log_content = job_logs.get(job["id"])
        if log_content:
            # Extract package versions (once per transformers version is enough)
            if version not in version_info:
                versions = parse_versions_from_log(log_content)
                if versions:
                    version_info[version] = versions

            test_results = parse_test_results_from_log(log_content)
            for (class_name, test_method), status in test_results.items():
                category = extract_test_category(class_name)
                model_name = extract_model_from_test_name(test_method)
                if model_name is None:
                    continue  # Skip unparseable test names
                model_key = (category, model_name)
                results[version][model_key][(runner, device)] = status

    # If no per-test results found, fall back to job-level conclusions
    if not any(results.values()):
        print("WARNING: Could not parse per-test results from logs. Using job-level conclusions only.",
              file=sys.stderr)
        for job in jobs:
            job_name = job["name"]
            version, device, runner = parse_job_name(job_name)
            if not version:
                continue
            conclusion = job.get("conclusion", "unknown")
            results[version][("Job", "all_tests")][(runner, device)] = conclusion

    # Generate markdown
    lines = []
    run_url = run_info.get("html_url", "")
    run_number = run_info.get("run_number", "")
    run_status = run_info.get("conclusion", run_info.get("status", "unknown"))
    head_branch = run_info.get("head_branch", "")
    created_at = run_info.get("created_at", "")

    lines.append("# OpenVINO GenAI AIPC Test Report")
    lines.append("")
    lines.append(f"**Workflow Run:** [#{run_number}]({run_url})")
    lines.append(f"**Branch:** `{head_branch}`")
    lines.append(f"**Status:** {run_status}")
    lines.append(f"**Date:** {created_at}")
    lines.append("")

    # Package versions table
    if version_info:
        lines.append("### Package Versions")
        lines.append("")
        lines.append("| Transformers | OpenVINO | OpenVINO GenAI | PyTorch |")
        lines.append("|:---:|:---:|:---:|:---:|")
        for tv in sorted(version_info.keys()):
            vi = version_info[tv]
            ov = vi.get("openvino", "-")
            ov_genai = vi.get("openvino_genai", "-")
            torch_ver = vi.get("torch", "-")
            tf_ver = vi.get("transformers", tv)
            lines.append(f"| {tf_ver} | {ov} | {ov_genai} | {torch_ver} |")
        lines.append("")

    # Status emoji mapping
    status_symbols = {
        "passed": "\u2705",
        "success": "\u2705",
        "failed": "\u274c",
        "failure": "\u274c",
        "skipped": "\u23ed\ufe0f",
        "error": "\u26a0\ufe0f",
        "cancelled": "\u23f9\ufe0f",
        "unknown": "\u2753",
    }

    runners = ["MTL", "LNL"]
    devices = ["CPU", "GPU", "NPU"]

    for version in sorted(results.keys()):
        lines.append(f"## Transformers {version}")
        lines.append("")

        # Table header with sub-columns
        header = "| Category | Model |"
        separator = "|----------|-------|"
        for runner in runners:
            for device in devices:
                header += f" {runner}/{device} |"
                separator += ":---:|"

        lines.append(header)
        lines.append(separator)

        # Group by category, sorted
        model_keys = sorted(results[version].keys(), key=lambda x: (x[0], x[1]))

        for category, model_name in model_keys:
            row = f"| {category} | {model_name} |"
            for runner in runners:
                for device in devices:
                    status = results[version][(category, model_name)].get((runner, device), "-")
                    symbol = status_symbols.get(status, status)
                    row += f" {symbol} |"
            lines.append(row)

        lines.append("")

    # Legend
    lines.append("## Legend")
    lines.append("")
    lines.append("| Symbol | Meaning |")
    lines.append("|--------|---------|")
    lines.append("| \u2705 | Passed |")
    lines.append("| \u274c | Failed |")
    lines.append("| \u23ed\ufe0f | Skipped |")
    lines.append("| \u26a0\ufe0f | Error (crash/access violation) |")
    lines.append("| \u23f9\ufe0f | Cancelled |")
    lines.append("| - | Not run / No data |")
    lines.append("")

    # Job-level summary
    lines.append("## Job-Level Summary")
    lines.append("")
    header = "| Version |"
    separator = "|---------|"
    for runner in runners:
        for device in devices:
            header += f" {runner}/{device} |"
            separator += ":---:|"
    lines.append(header)
    lines.append(separator)

    for version in sorted(set(v for v, _, _ in job_conclusions.keys())):
        row = f"| {version} |"
        for runner in runners:
            for device in devices:
                conclusion = job_conclusions.get((version, device, runner), "-")
                symbol = status_symbols.get(conclusion, conclusion)
                row += f" {symbol} |"
        lines.append(row)
    lines.append("")

    # Direct links to jobs
    lines.append("<details><summary>Job Links</summary>")
    lines.append("")
    lines.append("| Job | Conclusion |")
    lines.append("|-----|-----------|")
    for job in sorted(jobs, key=lambda j: j["name"]):
        job_name = job["name"]
        conclusion = job.get("conclusion", "unknown")
        symbol = status_symbols.get(conclusion, conclusion)
        job_url = job.get("html_url", "")
        lines.append(f"| [{job_name}]({job_url}) | {symbol} {conclusion} |")
    lines.append("")
    lines.append("</details>")
    lines.append("")

    return "\n".join(lines)


def markdown_to_html(md_text):
    """Convert markdown report to a self-contained HTML page.

    Handles: headings, bold, inline code, tables, links, <details> blocks, and paragraphs.
    Uses GitHub-like styling with no external dependencies.
    """
    html_lines = []

    html_lines.append("""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>OpenVINO GenAI AIPC Test Report</title>
<style>
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif; max-width: 1400px; margin: 0 auto; padding: 20px; color: #24292f; background: #fff; }
h1 { border-bottom: 1px solid #d0d7de; padding-bottom: 8px; }
h2 { border-bottom: 1px solid #d0d7de; padding-bottom: 6px; margin-top: 24px; }
h3 { margin-top: 20px; }
table { border-collapse: collapse; margin: 12px 0; width: auto; }
th, td { border: 1px solid #d0d7de; padding: 6px 12px; text-align: center; }
th { background: #f6f8fa; font-weight: 600; }
td:first-child, td:nth-child(2) { text-align: left; }
a { color: #0969da; text-decoration: none; }
a:hover { text-decoration: underline; }
code { background: #f6f8fa; padding: 2px 6px; border-radius: 3px; font-size: 90%; }
details { margin: 12px 0; }
summary { cursor: pointer; font-weight: 600; }
p { margin: 4px 0; }
</style>
</head>
<body>
""")

    in_table = False
    in_details = False
    lines = md_text.split("\n")

    for line in lines:
        # Details blocks (pass through)
        if line.strip().startswith("<details"):
            html_lines.append(line)
            in_details = True
            continue
        if line.strip() == "</details>":
            html_lines.append(line)
            in_details = False
            continue
        if line.strip().startswith("<summary"):
            html_lines.append(line)
            continue

        # Table
        if line.strip().startswith("|"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            # Skip separator rows
            if all(re.match(r"^[-:]+$", c) for c in cells):
                continue
            if not in_table:
                html_lines.append("<table>")
                # First row is header
                html_lines.append("<tr>" + "".join(f"<th>{_inline_md(c)}</th>" for c in cells) + "</tr>")
                in_table = True
            else:
                html_lines.append("<tr>" + "".join(f"<td>{_inline_md(c)}</td>" for c in cells) + "</tr>")
            continue
        else:
            if in_table:
                html_lines.append("</table>")
                in_table = False

        # Headings
        heading_match = re.match(r"^(#{1,6})\s+(.+)", line)
        if heading_match:
            level = len(heading_match.group(1))
            text = _inline_md(heading_match.group(2))
            html_lines.append(f"<h{level}>{text}</h{level}>")
            continue

        # Empty line
        if not line.strip():
            continue

        # Paragraph/text line
        html_lines.append(f"<p>{_inline_md(line)}</p>")

    if in_table:
        html_lines.append("</table>")

    html_lines.append("</body>\n</html>")
    return "\n".join(html_lines)


def _inline_md(text):
    """Convert inline markdown (bold, code, links) to HTML."""
    # Links: [text](url)
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', text)
    # Bold: **text**
    text = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", text)
    # Inline code: `text`
    text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
    return text


def main():
    parser = argparse.ArgumentParser(description="Generate markdown report for GenAI AIPC workflow run")
    parser.add_argument("--run-id", type=int, help="Specific workflow run ID (default: latest completed)")
    parser.add_argument("--output", "-o", type=str, help="Output file path (default: stdout)")
    parser.add_argument("--html", action="store_true", help="Also generate an HTML version of the report")
    parser.add_argument("--no-logs", action="store_true",
                        help="Skip downloading logs (only show job-level conclusions)")
    args = parser.parse_args()

    print("Fetching workflow run info...", file=sys.stderr)
    run_info = get_latest_run(args.run_id)
    run_id = run_info["id"]
    print(f"  Run #{run_info['run_number']} (ID: {run_id}), "
          f"status: {run_info.get('conclusion', run_info['status'])}",
          file=sys.stderr)

    print("Fetching jobs...", file=sys.stderr)
    jobs = get_jobs(run_id)
    print(f"  Found {len(jobs)} jobs", file=sys.stderr)

    job_logs = {}
    if not args.no_logs:
        print("Downloading job logs...", file=sys.stderr)
        for i, job in enumerate(jobs):
            job_name = job["name"]
            version, device, runner = parse_job_name(job_name)
            if not version:
                continue
            print(f"  [{i+1}/{len(jobs)}] {job_name}...", end="", file=sys.stderr)
            log = download_job_log(job["id"])
            if log:
                job_logs[job["id"]] = log
                print(" OK", file=sys.stderr)
            else:
                print(" no logs", file=sys.stderr)

    print("Generating report...", file=sys.stderr)
    report = build_report(run_info, jobs, job_logs)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"Report saved to {args.output}", file=sys.stderr)
    else:
        print(report)

    if args.html:
        html_content = markdown_to_html(report)
        if args.output:
            # Derive HTML filename from the markdown output path
            base, _ = os.path.splitext(args.output)
            html_path = base + ".html"
        else:
            html_path = "report.html"
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        print(f"HTML report saved to {html_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
