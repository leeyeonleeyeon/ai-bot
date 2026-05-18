"""GitHub Contents API committer — uses raw httpx, no PyGithub dependency.

repo: "owner/name", branch: 기본 main. 파일이 이미 있으면 sha를 받아 업데이트.
"""
from __future__ import annotations

import base64
import logging
import os
from typing import Optional

import httpx

log = logging.getLogger(__name__)

_API = "https://api.github.com"


class GitHubCommitter:
    def __init__(self, token: str, repo: str, branch: str = "main"):
        if "/" not in repo:
            raise ValueError(f"GITHUB_REPO must be 'owner/name', got {repo!r}")
        self.token = token
        self.repo = repo
        self.branch = branch

    async def commit_file(
        self,
        path: str,
        content: str,
        message: str,
    ) -> str:
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        url = f"{_API}/repos/{self.repo}/contents/{path}"
        payload: dict = {
            "message": message,
            "branch": self.branch,
            "content": base64.b64encode(content.encode("utf-8")).decode("ascii"),
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            existing = await client.get(
                url, headers=headers, params={"ref": self.branch}
            )
            if existing.status_code == 200:
                payload["sha"] = existing.json().get("sha")
            elif existing.status_code != 404:
                existing.raise_for_status()

            resp = await client.put(url, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()

        return (
            data.get("content", {}).get("html_url")
            or data.get("commit", {}).get("html_url")
            or ""
        )


def from_env() -> Optional[GitHubCommitter]:
    token = os.environ.get("GITHUB_TOKEN")
    repo = os.environ.get("GITHUB_REPO")
    branch = os.environ.get("GITHUB_BRANCH", "main")
    if not token or not repo:
        return None
    return GitHubCommitter(token=token, repo=repo, branch=branch)
