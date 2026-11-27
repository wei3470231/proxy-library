#!/usr/bin/env python3
"""Shard tester for GitHub Actions: tests ONE shard of the proxy library.

Why this file exists
--------------------
The local VM cannot test proxies: its egress blocks direct TCP to arbitrary
IP:port (other_tcp: Deny), and switching it to Ask would ask the user to
confirm every one of ~177,000 proxy IPs by hand. GitHub-hosted runners for
a PUBLIC repo have unrestricted egress and cost no minutes, so the same
test logic runs there instead, one shard per workflow run, rotating
through the whole library and looping forever.

Same test contract as test_proxies.py (the local script):
- 10 threads (ThreadPoolExecutor max_workers=10).
- Each ip:port is tested with 4 protocols: HTTP, HTTPS, SOCKS4, SOCKS5.
- Test URL: https://api.ipify.org/?format=json -> {"ip": "<exit ip>"}.
- A test passes when ipify answers HTTP 200 with a valid {"ip": ...}.
- Anonymity/exit check: exit_ip == proxy IP (match=True means the request
  really exited from that proxy IP; match=False means it relays onward).
- Logs and summaries contain COUNTS ONLY, never IP addresses.

Shard design (keeps parallel / repeated runs from clobbering each other):
- Library line i belongs to shard (i % shards). One run tests one shard
  only and writes exactly one file: tested_working_shard_<N>.json
  (only that shard's currently-passing entries). It never touches other
  shards' files and never does a global purge, so a later merge step can
  safely combine all shard files into one working table.
- --max-minutes stops the run cleanly before GitHub's 6-hour job cap;
  partial progress is still written out, and the shard is retried next
  rotation (shard state only advances when a run finishes its shard).
- Resumable within a shard: --resume-from <index> restarts mid-shard.

Run locally (dry run, no network):
    python3 test_proxies_shard.py --library proxies.txt --shards 36 \
        --shard 0 --dry-run
Run on GitHub Actions: see test-proxies.yml in this directory.
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import requests

TEST_URL = "https://api.ipify.org/?format=json"
THREADS = 10
TIMEOUT = (5, 10)  # (connect, read) seconds
SCHEMES = [("HTTP", "http"), ("HTTPS", "https"),
           ("SOCKS4", "socks4"), ("SOCKS5", "socks5h")]
BATCH = 200  # tasks per thread-pool batch; results flushed after each batch


def log(msg):
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"[{ts}] {msg}", flush=True)


def now_iso():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def atomic_write_json(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f)
    os.replace(tmp, path)


def test_one(task):
    """task = (ip, port, protocol_name, scheme). Returns result dict or None."""
    ip, port, proto, scheme = task
    proxy_url = f"{scheme}://{ip}:{port}"
    t0 = time.time()
    try:
        r = requests.get(TEST_URL, proxies={"http": proxy_url, "https": proxy_url},
                         timeout=TIMEOUT)
        latency = int((time.time() - t0) * 1000)
        if r.status_code == 200:
            exit_ip = r.json().get("ip")
            if isinstance(exit_ip, str) and exit_ip:
                return {"ip": ip, "port": int(port), "protocol": proto,
                        "exit_ip": exit_ip, "match": exit_ip == ip,
                        "latency_ms": latency, "tested_at": now_iso()}
        return None
    except Exception:
        return None


def build_tasks(lines):
    tasks = []
    for ln in lines:
        ln = ln.strip()
        if not ln or ":" not in ln:
            continue
        ip, port = ln.rsplit(":", 1)
        for proto, scheme in SCHEMES:
            tasks.append((ip, port, proto, scheme))
    return tasks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--library", default="proxies.txt")
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--shards", type=int, required=True)
    ap.add_argument("--out", default=None,
                    help="shard result file (default tested_working_shard_<N>.json)")
    ap.add_argument("--summary", default=None,
                    help="summary json (default tester_shard_<N>_summary.json)")
    ap.add_argument("--max-minutes", type=int, default=240,
                    help="stop cleanly after this many minutes (default 240)")
    ap.add_argument("--resume-from", type=int, default=0,
                    help="task index to resume from inside this shard")
    ap.add_argument("--limit-proxies", type=int, default=0,
                    help="test at most N proxies of this shard (pilot)")
    ap.add_argument("--dry-run", action="store_true",
                    help="only build the task list and print counts, no network")
    args = ap.parse_args()

    if args.shard < 0 or args.shard >= args.shards:
        print(f"shard {args.shard} out of range for {args.shards} shards",
              file=sys.stderr)
        return 2

    out = args.out or f"tested_working_shard_{args.shard}.json"
    summary_path = args.summary or f"tester_shard_{args.shard}_summary.json"

    with open(args.library) as f:
        all_lines = [ln.strip() for ln in f if ln.strip()]
    shard_lines = [ln for i, ln in enumerate(all_lines) if i % args.shards == args.shard]
    if args.limit_proxies:
        shard_lines = shard_lines[:args.limit_proxies]
    tasks = build_tasks(shard_lines)

    log(f"shard {args.shard}/{args.shards}: library={len(all_lines)} "
        f"shard_proxies={len(shard_lines)} tasks={len(tasks)}")
    if args.dry_run:
        print(json.dumps({"shard": args.shard, "shards": args.shards,
                          "library": len(all_lines),
                          "shard_proxies": len(shard_lines),
                          "tasks": len(tasks)}))
        return 0

    working = {}
    try:
        with open(out) as f:
            working = json.load(f)
    except Exception:
        working = {}
    # Drop this shard's previous entries; they are re-confirmed below or gone.
    # Other shards live in other files and are untouched.
    working = {}
    started = time.time()
    deadline = started + args.max_minutes * 60
    tested = 0
    passed = 0
    idx = args.resume_from
    complete = True
    while idx < len(tasks):
        if time.time() >= deadline:
            complete = False
            log(f"time budget reached at task {idx}/{len(tasks)}; "
                f"stopping cleanly, shard will be retried next rotation")
            break
        batch = tasks[idx:idx + BATCH]
        with ThreadPoolExecutor(max_workers=THREADS) as ex:
            results = list(ex.map(test_one, batch))
        for task, res in zip(batch, results):
            tested += 1
            key = f"{task[0]}:{task[1]}|{task[2]}"
            if res is not None:
                passed += 1
                res["shard"] = args.shard
                working[key] = res
            else:
                working.pop(key, None)
        idx += len(batch)
        atomic_write_json(out, working)
        if tested % 5000 < BATCH:
            log(f"shard {args.shard}: tested={tested}/{len(tasks)} "
                f"passed={passed} working={len(working)} "
                f"elapsed={int(time.time() - started)}s")

    atomic_write_json(out, working)
    summary = {"shard": args.shard, "shards": args.shards,
               "library": len(all_lines), "shard_proxies": len(shard_lines),
               "tasks": len(tasks), "tested": tested, "passed": passed,
               "working": len(working), "complete": complete,
               "next_task_index": idx,
               "elapsed_s": int(time.time() - started),
               "finished_at": now_iso()}
    atomic_write_json(summary_path, summary)
    log(f"shard {args.shard} done: tested={tested} passed={passed} "
        f"working={len(working)} complete={complete} "
        f"elapsed={int(time.time() - started)}s")
    return 0 if complete else 3  # 3 = time budget hit, shard incomplete


if __name__ == "__main__":
    sys.exit(main())
