from __future__ import annotations

import io
import json
import time
import zipfile
from dataclasses import dataclass

import httpx
import jwt

from app.config import Settings


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

    async def _request_bytes(
        self,
        method: str,
        path: str,
        *,
        installation_id: int | None,
    ) -> bytes:
        token = await self._installation_token(installation_id)
        async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
            response = await client.request(
                method,
                f"{self.base_url}{path}",
                headers=self._headers(token),
            )
        self._raise_for_status(response)
        return response.content

    def _raise_for_status(self, response: httpx.Response) -> None:
        if response.status_code < 400:
            return
        raise GitHubError(
            f"GitHub API {response.status_code} {response.request.method} "
            f"{response.request.url}: {response.text}"
        )

    async def get_pull(self, repo: str, pr_number: int, installation_id: int | None) -> dict:
        result = await self._request(
            "GET",
            f"/repos/{repo}/pulls/{pr_number}",
            installation_id=installation_id,
        )
        assert isinstance(result, dict)
        return result

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

    async def find_workflow_run_by_dispatch_id(
        self,
        *,
        repo: str,
        installation_id: int | None,
        dispatch_id: str,
    ) -> int | None:
        data = await self._request(
            "GET",
            f"/repos/{repo}/actions/runs",
            installation_id=installation_id,
            params={"per_page": 20},
        )
        if not isinstance(data, dict):
            return None
        for run in data.get("workflow_runs", []):
            name = str(run.get("name") or "")
            if dispatch_id in name:
                return int(run["id"])
        return None

    async def get_scheduler_outputs(
        self,
        *,
        repo: str,
        installation_id: int | None,
        workflow_run_id: int,
        dispatch_id: str,
    ) -> dict[str, str]:
        data = await self._request(
            "GET",
            f"/repos/{repo}/actions/runs/{workflow_run_id}/artifacts",
            installation_id=installation_id,
            params={"name": dispatch_id, "per_page": 100},
        )
        if not isinstance(data, dict):
            raise GitHubError("GitHub returned an invalid artifact listing")

        matches = [
            artifact
            for artifact in data.get("artifacts", [])
            if artifact.get("name") == dispatch_id and not artifact.get("expired", False)
        ]
        if len(matches) != 1:
            raise GitHubError(
                f"expected one scheduler output artifact for {dispatch_id}, found {len(matches)}"
            )

        archive = await self._request_bytes(
            "GET",
            f"/repos/{repo}/actions/artifacts/{int(matches[0]['id'])}/zip",
            installation_id=installation_id,
        )
        if len(archive) > 256 * 1024:
            raise GitHubError("scheduler output artifact exceeds 256 KiB compressed limit")

        try:
            with zipfile.ZipFile(io.BytesIO(archive)) as bundle:
                files = [
                    info
                    for info in bundle.infolist()
                    if not info.is_dir() and info.filename.rsplit("/", 1)[-1] == "scheduler-outputs.json"
                ]
                if len(files) != 1:
                    raise GitHubError(
                        "scheduler output artifact must contain exactly one scheduler-outputs.json"
                    )
                if files[0].file_size > 64 * 1024:
                    raise GitHubError("scheduler-outputs.json exceeds 64 KiB limit")
                raw = bundle.read(files[0])
        except zipfile.BadZipFile as exc:
            raise GitHubError("scheduler output artifact is not a valid zip archive") from exc

        try:
            payload = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise GitHubError("scheduler-outputs.json is not valid UTF-8 JSON") from exc
        if not isinstance(payload, dict) or payload.get("version") != 1:
            raise GitHubError("unsupported scheduler output schema version")
        outputs = payload.get("outputs")
        if not isinstance(outputs, dict) or len(outputs) > 64:
            raise GitHubError("scheduler outputs must be an object with at most 64 entries")

        validated: dict[str, str] = {}
        for key, value in outputs.items():
            if not isinstance(key, str) or not key or len(key) > 128:
                raise GitHubError("scheduler output names must be 1-128 character strings")
            if not isinstance(value, str) or len(value.encode("utf-8")) > 16 * 1024:
                raise GitHubError(
                    f"scheduler output {key!r} must be a string no larger than 16 KiB"
                )
            validated[key] = value
        return validated
