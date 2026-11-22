# SPDX-License-Identifier: Apache-2.0
"""Local SQLite ledger for review reports and spend.

One review runs at a time. Concurrent budget reservation is a later phase.
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
            "SELECT report_json FROM runs WHERE review_input_sha256 = ?",
            (review_input_sha256,),
        ).fetchone()
        if row is None:
            return None
        else:
            return Report.model_validate_json(row[0])

    def store(
        self, pull_request_number: int, report: Report, reviewed_at: datetime
    ) -> None:
        self.connection.execute(
            """
            INSERT INTO runs (
                review_input_sha256,
                pull_request_number,
                report_json,
                cost_cents,
                created_at
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                report.review_input_sha256,
                pull_request_number,
                report.model_dump_json(),
                report.usage_cost_cents,
                reviewed_at.isoformat(),
            ),
        )
        self.connection.commit()

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
    ) -> SpendAuthorization:
        rounds = self.round_count(pull_request_number)
        worst_case_cents = (retry_count + 1) * budget.call_cost_cents
        if rounds >= 2:
            return SpendAuthorization(
                allowed=False, reason="pull request round limit reached"
            )
        elif rounds == 1 and not follow_up_authorized:
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
