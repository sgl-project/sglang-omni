// Package archtest holds no code, only tests of the code's structure, run
// with every go test ./... (docs/architecture.md, "Code structure rules"):
//
//   - TestFileSizes: a Go file has at most 600 lines, a test file 800; 400
//     is the aim for code, reported (go test -v) but not failed. Files over
//     the limit today are in allowlist.txt with a ceiling they may not grow
//     past (a ratchet), and why and when they will be split; split them and
//     remove their lines. Generated files (// Code generated … DO NOT
//     EDIT.), third_party, testdata, the build output and the Swift helper
//     are not counted.
//   - TestImports: the layering of internal/ (imports_test.go has the rules
//     and why), and every import of one internal package by another listed
//     in imports.txt, so that a new dependency shows in review.
//   - TestLongFuncs: reports (t.Log, go test -v) functions over 150 lines.
//   - TestNotice: every module go.mod requires is named in NOTICE with its
//     version and has its licence text in LICENSES/ (AGENTS.md, "Licence").
//
// Each failure says how to fix it.
package archtest
