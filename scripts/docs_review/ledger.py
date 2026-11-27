# SPDX-License-Identifier: Apache-2.0
"""Local SQLite ledger for review reports and spend.

Costs are estimates reserved before each model call.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path

from scripts.docs_review.models import BudgetConfig, Report, SpendAuthorization


class ReviewLedger:
    """Stores one report per input hash and the spend for each pull request."""

    def __init__(self, ledger_path: Path) -> None:
        ledger_path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(ledger_path)
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS runs (
                review_input_sha256 TEXT PRIMARY KEY,
                pull_request_number INTEGER NOT NULL,
                report_json TEXT NOT NULL,
                cost_cents INTEGER NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()

    def fetch(self, review_input_sha256: str) -> Report | None:
        row = self.connection.execute(
            "SELECT report_json FROM runs WHERE review_input_sha256 = ? AND report_json != ''",
            (review_input_sha256,),
        ).fetchone()
        if row is None:
            return None
        else:
            return Report.model_validate_json(row[0])

    def reserve_attempt(
        self,
        reservation_id: str,
        first_attempt: bool,
        pull_request_number: int,
        budget: BudgetConfig,
        follow_up_authorized: bool,
        remaining_retries: int,
        reviewed_at: datetime,
    ) -> SpendAuthorization:
        with self.connection:
            self.connection.execute("BEGIN IMMEDIATE")
            authorization = self.authorize(
                pull_request_number,
                budget,
                follow_up_authorized,
                remaining_retries,
                reviewed_at,
                first_attempt,
            )
            if not authorization.allowed:
                return authorization
            elif first_attempt:
                self.connection.execute(
                    "INSERT INTO runs (review_input_sha256, pull_request_number, report_json, cost_cents, created_at) VALUES (?, ?, '', ?, ?)",
                    (
                        reservation_id,
                        pull_request_number,
                        budget.call_cost_cents,
                        reviewed_at.isoformat(),
                    ),
                )
            else:
                self.connection.execute(
                    "UPDATE runs SET cost_cents = cost_cents + ? WHERE review_input_sha256 = ?",
                    (budget.call_cost_cents, reservation_id),
                )
        return authorization

    def store(self, reservation_id: str, report: Report) -> None:
        with self.connection:
            self.connection.execute("BEGIN IMMEDIATE")
            cache_key = report.review_input_sha256
            if self.fetch(cache_key) is not None:
                cache_key = reservation_id
            else:
                pass
            self.connection.execute(
                "UPDATE runs SET review_input_sha256 = ?, report_json = ? WHERE review_input_sha256 = ?",
                (cache_key, report.model_dump_json(), reservation_id),
            )

    def pull_request_spend_cents(self, pull_request_number: int) -> int:
        row = self.connection.execute(
            "SELECT COALESCE(SUM(cost_cents), 0) FROM runs WHERE pull_request_number = ?",
            (pull_request_number,),
        ).fetchone()
        return int(row[0])

    def round_count(self, pull_request_number: int) -> int:
        row = self.connection.execute(
            "SELECT COUNT(*) FROM runs WHERE pull_request_number = ?",
            (pull_request_number,),
        ).fetchone()
        return int(row[0])

    def day_spend_cents(self, reviewed_at: datetime) -> int:
        prefix = f"{reviewed_at.date().isoformat()}%"
        row = self.connection.execute(
            "SELECT COALESCE(SUM(cost_cents), 0) FROM runs WHERE created_at LIKE ?",
            (prefix,),
        ).fetchone()
        return int(row[0])

    def authorize(
        self,
        pull_request_number: int,
        budget: BudgetConfig,
        follow_up_authorized: bool,
        retry_count: int,
        reviewed_at: datetime,
        first_attempt: bool = True,
    ) -> SpendAuthorization:
        rounds = self.round_count(pull_request_number)
        worst_case_cents = (retry_count + 1) * budget.call_cost_cents
        if first_attempt and rounds >= 2:
            return SpendAuthorization(
                allowed=False, reason="pull request round limit reached"
            )
        elif first_attempt and rounds == 1 and not follow_up_authorized:
            return SpendAuthorization(
                allowed=False, reason="follow-up review is not authorized"
            )
        elif budget.call_cost_cents > budget.max_cost_cents_per_call:
            return SpendAuthorization(
                allowed=False, reason="call cost exceeds the per-call budget"
            )
        elif (
            self.pull_request_spend_cents(pull_request_number) + worst_case_cents
            > budget.max_cost_cents_per_pull_request
        ):
            return SpendAuthorization(
                allowed=False, reason="pull request budget exceeded"
            )
        elif (
            self.day_spend_cents(reviewed_at) + worst_case_cents
            > budget.max_cost_cents_per_day
        ):
            return SpendAuthorization(allowed=False, reason="daily budget exceeded")
        else:
            return SpendAuthorization(allowed=True, reason="")
