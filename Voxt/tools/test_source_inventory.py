import unittest

from audit_source_inventory import inventory


class SourceInventoryTests(unittest.TestCase):
    def testSingleOccurrenceIsOnlyACandidateAndReferencesSuppressIt(self):
        report = inventory({"Voxt/A.swift": "func unused() {}\nfunc used() {}\nused()\n"})
        self.assertEqual([item["name"] for item in report["single_occurrence_candidates"]], ["unused"])
        self.assertIn("no reachability", report["warning"])

    def testLocationsDoNotExposeCredentialOrTranscriptValues(self):
        report = inventory({"Voxt/A.swift": 'VoxtLog.info("token=secret_fixture")\nlet x = try! Data(contentsOf: url)\n'})
        findings = report["review_locations"]
        self.assertEqual(findings["blocking_file_read"], [{"path": "Voxt/A.swift", "line": 2}])
        self.assertEqual(findings["credential_or_content_log_review"], [{"path": "Voxt/A.swift", "line": 1}])
        self.assertNotIn("secret_fixture", str(report))

    def testImportsSeparateAppAndTestUsersWithoutDeclaringDependenciesUnused(self):
        report = inventory({"Voxt/A.swift": "@preconcurrency import MLX\n", "VoxtTests/A.swift": "import XCTest\nimport MLX\n"})
        self.assertEqual(report["import_users"]["MLX"], ["Voxt/A.swift", "VoxtTests/A.swift"])
        self.assertEqual(report["import_users"]["XCTest"], ["VoxtTests/A.swift"])
        self.assertEqual(report["file_count"], 2)

    def testAsyncAndTimerSitesAreCollectedWithoutClaimingLeaks(self):
        report = inventory({"Voxt/A.swift": "Task { await work() }\nTask.detached { work() }\nTimer.publish(every: 1)\n"})
        self.assertEqual(len(report["review_locations"]["unstructured_task"]), 2)
        self.assertEqual(len(report["review_locations"]["repeating_timer"]), 1)


if __name__ == "__main__":
    unittest.main()
