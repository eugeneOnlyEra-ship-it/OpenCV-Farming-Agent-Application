#!/usr/bin/env python3
"""Copy aws/lambda/farm_api.py into aws/farm_api.yaml as the Lambda's inline code (ZipFile).

CloudFormation lets a Lambda's code be written straight into the template (up to 4096 characters), so deploying the API
needs no zip, no S3 bucket and no SAM. Run this after every edit to aws/lambda/farm_api.py:   python aws/sync_inline_code.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
LIMIT = 4096


def inline_block(code, indent="          "):
    return "".join((indent + line if line.strip() else "") + "\n" for line in code.rstrip("\n").split("\n"))


def sync(check_only=False):
    code = open(os.path.join(HERE, "lambda", "farm_api.py"), encoding="utf-8").read()
    if len(code) > LIMIT:
        sys.exit(f"farm_api.py is {len(code)} characters; CloudFormation's inline limit is {LIMIT}. Shorten it.")
    path = os.path.join(HERE, "farm_api.yaml")
    lines = open(path, encoding="utf-8").read().split("\n")
    i = next(n for n, ln in enumerate(lines) if ln.strip() == "ZipFile: |")
    j = i + 1
    while j < len(lines) and (lines[j].startswith("          ") or not lines[j].strip()):
        j += 1
    # keep the blank line that separates this resource from the next one
    k = j
    while k > i + 1 and not lines[k - 1].strip():
        k -= 1
    new = lines[:i + 1] + inline_block(code).rstrip("\n").split("\n") + [""] + lines[k:] if k == j else \
        lines[:i + 1] + inline_block(code).rstrip("\n").split("\n") + lines[k:]
    text = "\n".join(new)
    if check_only:
        return text == "\n".join(lines)
    open(path, "w", encoding="utf-8").write(text)
    print(f"farm_api.yaml now carries farm_api.py inline ({len(code)} of {LIMIT} characters)")
    return True


if __name__ == "__main__":
    sync()
