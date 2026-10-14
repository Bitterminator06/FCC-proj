"""
Helper script to configure your remote repository and push code.
Usage:
  python git_push.py <REMOTE_URL> [BRANCH]
Example:
  python git_push.py https://github.com/your-username/asl-cloud-recognizer.git
"""

import sys
import os
from dulwich.repo import Repo
from dulwich.porcelain import add, commit, push

def main():
    if len(sys.argv) < 2:
        print("Usage: python git_push.py <REMOTE_GIT_URL> [BRANCH]")
        print("Example: python git_push.py https://github.com/username/repo.git main")
        sys.exit(1)

    remote_url = sys.argv[1]
    branch = sys.argv[2] if len(sys.argv) > 2 else "main"

    repo = Repo(".")
    print(f"[Git] Staging all files...")
    add(".", ".")

    try:
        c = commit(".", message=b"Update: ASL Alphabet Recognizer edge-cloud code and benchmarks")
        print(f"[Git] Created commit: {c.decode('ascii') if isinstance(c, bytes) else c}")
    except Exception as e:
        print(f"[Git] Commit info: {e} (Working directory may already be clean)")

    print(f"[Git] Pushing to remote: {remote_url} (branch: {branch})...")
    try:
        push(".", remote_url, refspecs=[f"HEAD:refs/heads/{branch}".encode("ascii")])
        print(f"[Git] Successfully pushed to {remote_url}!")
    except Exception as e:
        print(f"[Git Error] Push failed: {e}")
        print("\nTip: If authentication is required, use a Personal Access Token (PAT):")
        print(f"  python git_push.py https://<TOKEN>@github.com/<USER>/<REPO>.git {branch}")

if __name__ == "__main__":
    main()
