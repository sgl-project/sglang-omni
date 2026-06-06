from __future__ import annotations

import time
from dataclasses import dataclass

import httpx
import jwt

from .config import Settings


class GitHubError(RuntimeError):
    pass


@dataclass(frozen=True)
class CheckRun:
    id: int
    name: str
    status: str
    conclusion: str | None


class GitHubClient:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.base_url = "https://api.github.com"

    def _app_jwt(self) -> str:
        private_key = self.settings.github_private_key
        if not self.settings.app_id or not private_key:
            raise GitHubError("GitHub App credentials are not configured")
        now = int(time.time())
        payload = {
            "iat": now - 60,
            "exp": now + 9 * 60,
            "iss": self.settings.app_id,
        }
        return jwt.encode(payload, private_key, algorithm="RS256")

    async def _installation_token(self, installation_id: int | None) -> str:
        if self.settings.fallback_token:
            return self.settings.fallback_token
        if installation_id is None:
            raise GitHubError("installation_id is required without GITHUB_TOKEN fallback")
        headers = self._headers(self._app_jwt())
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(
                f"{self.base_url}/app/installations/{installation_id}/access_tokens",
                headers=headers,
            )
        self._raise_for_status(response)
        return response.json()["token"]

    def _headers(self, token: str) -> dict[str, str]:
        return {
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": "2022-11-28",
        }

    async def _request(
        self,
        method: str,
        path: str,
        *,
        installation_id: int | None,
        json: dict | None = None,
        params: dict | None = None,
    ) -> dict | list | None:
        token = await self._installation_token(installation_id)
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.request(
                method,
                f"{self.base_url}{path}",
                headers=self._headers(token),
                json=json,
                params=params,
            )
        self._raise_for_status(response)
        if not response.content:
            return None
        return response.json()

    def _raise_for_status(self, response: httpx.Response) -> None:
        if response.status_code < 400:
            return
        raise GitHubError(
            f"GitHub API {response.status_code} {response.request.method} "
            f"{response.request.url}: {response.text}"
        )

    async def get_pull(self, repo: str, pr_number: int, installation_id: int | None) -> dict:
        return await self._request(
            "GET",
            f"/repos/{repo}/pulls/{pr_number}",
            installation_id=installation_id,
        )

    async def create_check_run(
        self,
        *,
        repo: str,
        installation_id: int | None,
        name: str,
        head_sha: str,
        status: str,
        external_id: str,
        output_title: str,
        output_summary: str,
    ) -> CheckRun:
        data = await self._request(
            "POST",
            f"/repos/{repo}/check-runs",
            installation_id=installation_id,
            json={
                "name": name,
                "head_sha": head_sha,
                "status": status,
                "external_id": external_id,
                "output": {
                    "title": output_title,
                    "summary": output_summary,
                },
            },
        )
        assert isinstance(data, dict)
        return CheckRun(
            id=int(data["id"]),
            name=data["name"],
            status=data["status"],
            conclusion=data.get("conclusion"),
        )

    async def update_check_run(
        self,
        *,
        repo: str,
        installation_id: int | None,
        check_run_id: int,
        status: str,
        conclusion: str | None = None,
        output_title: str | None = None,
        output_summary: str | None = None,
    ) -> None:
        body: dict = {"status": status}
        if conclusion is not None:
            body["conclusion"] = conclusion
        if output_title or output_summary:
            body["output"] = {
                "title": output_title or "CI stage",
                "summary": output_summary or "",
            }
        await self._request(
            "PATCH",
            f"/repos/{repo}/check-runs/{check_run_id}",
            installation_id=installation_id,
            json=body,
        )

    async def dispatch_workflow(
        self,
        *,
        repo: str,
        installation_id: int | None,
        workflow_id: str,
        ref: str,
        inputs: dict[str, str],
    ) -> None:
        await self._request(
            "POST",
            f"/repos/{repo}/actions/workflows/{workflow_id}/dispatches",
            installation_id=installation_id,
            json={"ref": ref, "inputs": inputs},
        )
