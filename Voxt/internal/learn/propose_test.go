package learn

import (
	"slices"
	"strings"
	"testing"
)

// voxt1xMonitorTests is every case of Voxt 1.x's
// AutomaticDictionaryLearningMonitorTests.swift and where it went: "" means
// ported, to the test named in the row or in a table of this file or of
// watch_test.go; a reason means it does not apply here.
var voxt1xMonitorTests = []struct{ name, where, reason string }{
	{name: "testAutoKeyPressSkipsObservationScheduling", where: "TestLearnNeverReads"},
	{name: "testTranscriptionWithoutAutoKeyPressSchedulesObservation", where: "TestLearnNeverReads"},
	{name: "testBuildsLearningRequestForInPlaceCorrection", where: "TestJudgeEdit"},
	{name: "testBuildsLearningRequestForEqualLengthReplacement", where: "TestJudgeEdit"},
	{name: "testBuildsLearningRequestForAdjacentEnglishPhraseCorrection", where: "TestJudgeEdit"},
	{name: "testBuildsLearningRequestForMultiClauseChineseCorrectionWithoutMergingWholeSentence", where: "TestJudgeEdit"},
	{name: "testBuildsLearningRequestForDomainCorrectionParagraphs", where: "TestJudgeEditDomainParagraphs"},
	{name: "testDirectCandidateTermsFallbackReturnsFinalCorrectedToolName", where: "TestDirectTerm"},
	{name: "testSkipsDeletionOnlyChangeWhenBaselineWrapsInsertedTextAcrossLines", where: "TestJudgeEdit"},
	{name: "testBuildsLearningRequestFromTerminalLineWhenFinalSnapshotContainsCommandOutput", where: "TestJudgeEdit"},
	{name: "testObservationScopedTextPrefersEchoedCommandLineAfterPromptClears", where: "TestScopedCurrent"},
	{name: "testObservationSettlesAfterEchoedCommandContainsCompletedReplacement", where: "TestIncompleteEdit"},
	{name: "testBuildsLearningRequestWithExpandedTokenBoundaryFragments", where: "TestJudgeEdit"},
	{name: "testBuildsLearningRequestByComparingInsertedTextAgainstFinalScopedLine", where: "TestJudgeEdit (the contexts it also asserts were the language model's input, which is gone)"},
	{name: "testContinuesObservingWhenLatestEditLooksLikeDeletionOnly", where: "TestIncompleteEdit"},
	{name: "testDoesNotContinueObservingWhenReplacementTextAlreadyExists", where: "TestIncompleteEdit"},
	{name: "testObservationStopsAfterConsecutiveMissingSnapshotsBeforeAnyEdit", where: "TestWatchStateMissing"},
	{name: "testObservationSettlesAfterConsecutiveMissingSnapshotsOnceEditIsIdle", where: "TestWatchStateMissing"},
	{name: "testObservationDoesNotSettleAfterMissingSnapshotsBeforeIdleThreshold", where: "TestWatchStateMissing"},
	{name: "testObservationDoesNotSettleAfterMissingSnapshotsWhenReplacementStillIncomplete", where: "TestWatchStateMissing"},
	{name: "testObservationContinuesWhenIdleSnapshotStillLooksLikeDeletionOnly", where: "TestWatchStateFocused"},
	{name: "testObservationContinuesWhenIdleSnapshotStillContainsUnfinishedDeletionGroup", where: "TestWatchStateFocused"},
	{name: "testObservationSettlesWhenIdleSnapshotContainsCompletedReplacement", where: "TestWatchStateFocused"},
	{name: "testStableFocusedEditDoesNotFinalizeObservationImmediately", where: "TestWatch (edit at second 3, focus kept)"},
	{name: "testObservationResetsMissingCounterWhenFocusedSnapshotReturns", where: "TestWatchStateFocused"},
	{name: "testObservationSettlesAfterDeletionIntermediateWhenReplacementIsCompleted", where: "TestWatchStateFocused"},
	{name: "testObservationRegistersNewInputChangeAndResetsMissingCounter", where: "TestWatchStateFocused"},
	{name: "testRejectsPureAppendAfterInsertion", where: "TestJudgeEdit"},
	{name: "testRejectsUnrelatedEditsOutsideInsertedText", where: "TestJudgeEdit"},
	{name: "testRejectsLargeRewrite", where: "TestJudgeEdit"},
	{name: "testBuildPromptResolvesTemplateVariables", reason: "language model prompt: learning is local rules only (decision Q3)"},
	{name: "testBuildPromptUsesEmptyPlaceholderForExistingTerms", reason: "language model prompt (Q3)"},
	{name: "testBuildPromptIncludesExtractedCandidateTerms", reason: "language model prompt (Q3)"},
	{name: "testBuildPromptCapsExistingTermListToTwentyItems", reason: "language model prompt (Q3)"},
}

func TestEveryMonitorTestAccountedFor(t *testing.T) {
	if len(voxt1xMonitorTests) != 34 {
		t.Fatalf("%d cases, want 34", len(voxt1xMonitorTests))
	}
	seen := map[string]bool{}
	for _, c := range voxt1xMonitorTests {
		if seen[c.name] || (c.where == "") == (c.reason == "") {
			t.Errorf("bad row %+v", c)
		}
		seen[c.name] = true
	}
}

func TestJudgeEdit(t *testing.T) {
	for _, tc := range []struct {
		name                    string // the 1.x test ported, or the card's case
		inserted, baseline, end string
		before, after           string
		skip                    string // a part of the reason nothing is proposed
	}{
		{name: "testBuildsLearningRequestForInPlaceCorrection",
			inserted: "anthropic ai", baseline: "Please ship anthropic ai today.", end: "Please ship Anthropic today.",
			before: "anthropic ai", after: "Anthropic"},
		{name: "testBuildsLearningRequestForEqualLengthReplacement",
			inserted: "Waxed", baseline: "Our app is named Waxed.", end: "Our app is named Voxt.",
			before: "Waxed.", after: "Voxt."},
		{name: "testBuildsLearningRequestForAdjacentEnglishPhraseCorrection",
			inserted: "你好，您来帮我看一下我们新的公众号里面的 code code 有什么文章。",
			baseline: "你好，您来帮我看一下我们新的公众号里面的 code code 有什么文章。",
			end:      "你好，您来帮我看一下我们新的公众号里面的 claude code 有什么文章。",
			before:   "code", after: "claude"},
		{name: "testBuildsLearningRequestForMultiClauseChineseCorrectionWithoutMergingWholeSentence",
			inserted: "看一下我们投坑中有没有新的词源了。这个新的词源也需要接飞。",
			baseline: "看一下我们投坑中有没有新的词源了。这个新的词源也需要接飞。",
			end:      "看一下我们投坑中有没有新的词元了。这个新的词元也需要接入。",
			before:   "看一下我们投坑中有没有新的词源了。这个新的词源也需要接飞",
			after:    "看一下我们投坑中有没有新的词元了。这个新的词元也需要接入"},
		{name: "testBuildsLearningRequestFromTerminalLineWhenFinalSnapshotContainsCommandOutput",
			inserted: "你帮我看一下我们的 Go Host 能不能识别我们现在 APP 中的内容呀？",
			baseline: "~/x/doit/voxt-service\n> 你帮我看一下我们的 Go Host 能不能识别我们现在 APP 中的内容呀？",
			end:      "> 你帮我看一下我们的 Ghostty 能不能识别我们现在 APP 中的内容呀？\nzsh: command not found: 你帮我看一下我们的\n\n~/x/doit/voxt-service",
			before:   "Go Host", after: "Ghostty"},
		{name: "testBuildsLearningRequestWithExpandedTokenBoundaryFragments",
			inserted: "你帮我识别一下 WeChat 中的输入文本，我们是支持的吗？还有我们要支持一下 SG 狼魔鬼穷的文本查询。",
			baseline: "你帮我识别一下 WeChat 中的输入文本，我们是支持的吗？还有我们要支持一下 SG 狼魔鬼穷的文本查询。",
			end:      "你帮我识别一下 WeChat 中的输入文本，我们是支持的吗？还有我们要支持一下 SGLang 魔鬼群的文本查询。",
			before:   "狼魔鬼穷", after: "SGLang 魔鬼群的文本查询"},
		{name: "testBuildsLearningRequestByComparingInsertedTextAgainstFinalScopedLine",
			inserted: "你好，你帮我们看一下 WeChat 中的 SG 骆魔鬼群，他们用户在说什么？",
			baseline: "> 你帮我看一下我们的 Ghostty 能不能识别我们现在 APP 中的内容呀？\nzsh: command not found: 你帮我看一下我们的\n\n ~/x/doit/voxt-service  main !20 ?9\n> 你好，你帮我们看一下 WeChat 中的 SG 骆魔鬼群，他们用户在说什么？",
			end:      "> 你帮我看一下我们的 Ghostty 能不能识别我们现在 APP 中的内容呀？\nzsh: command not found: 你帮我看一下我们的\n> 你好，你帮我们看一下 WeChat 中的 SGLang魔鬼群，他们用户在说什么？\nzsh: command not found: 你好，你帮我们看一下\n\n ~/x/doit/voxt-service  main !20 ?9\n>",
			before:   "骆", after: "SGLang魔鬼群"},
		{name: "testSkipsDeletionOnlyChangeWhenBaselineWrapsInsertedTextAcrossLines",
			inserted: "React JS 和 Next JS",
			baseline: "我们使用了 React JS\n和 Next JS 来实现整个 APP 的链路。",
			end:      "我们使用了 React 和 Next JS 来实现整个 APP 的链路。",
			skip:     "no meaningful terms"},
		{name: "testRejectsPureAppendAfterInsertion",
			inserted: "hello", baseline: "hello", end: "hello world", skip: "does not intersect"},
		{name: "testRejectsUnrelatedEditsOutsideInsertedText",
			inserted: "Anthropic", baseline: "Anthropic works. tomorrow 3pm", end: "Anthropic works. tomorrow 4pm", skip: "does not intersect"},
		{name: "testRejectsLargeRewrite",
			inserted: "short note", baseline: "short note",
			end: "Completely different long paragraph with multiple rewritten clauses and unrelated content.", skip: "edit ratio"},
		{name: "nothing changed", inserted: "hello", baseline: "hello there", end: "hello there", skip: "identical"},
		{name: "scope order: inserted line's own skip is final, the whole text is not tried",
			inserted: "dictated line", baseline: "other 1\ndictated line\nother 2", end: "other 1 Edited\ndictated line\nother 2", skip: "no meaningful terms"},
		{name: "scope order: inserted text not in the baseline, the whole text is compared and its skip stands",
			inserted: "dictated line", baseline: "something else entirely", end: "something else Entirely", skip: "inserted text not found"},
		{name: "scope order: inserted line wins over the rest of the text",
			inserted: "dictated Wechart line", baseline: "other 1 ok\ndictated Wechart line", end: "other 1 changed Here\ndictated WeChat line",
			before: "Wechart", after: "WeChat"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got := judgeEdit(tc.inserted, tc.baseline, tc.end)
			if tc.skip != "" {
				if !strings.Contains(got.skip, tc.skip) {
					t.Fatalf("skip %q, want %q", got.skip, tc.skip)
				}
				return
			}
			if got.skip != "" {
				t.Fatalf("skipped: %s", got.skip)
			}
			if got.before != tc.before || got.after != tc.after {
				t.Fatalf("got %q → %q, want %q → %q", got.before, got.after, tc.before, tc.after)
			}
			if got.ratio > maxEditRatio {
				t.Errorf("edit ratio %.3f", got.ratio)
			}
		})
	}
}

func TestJudgeEditDomainParagraphs(t *testing.T) {
	overlap := func(actual, expected string) bool {
		return strings.Contains(actual, expected) || strings.Contains(expected, actual)
	}
	for _, tc := range domainParagraphs {
		t.Run(tc.name, func(t *testing.T) {
			end := strings.ReplaceAll(tc.inserted, tc.before, tc.after)
			if end == tc.inserted {
				t.Fatal("fixture target missing")
			}
			got := judgeEdit(tc.inserted, tc.inserted, end)
			if got.skip != "" {
				t.Fatalf("skipped: %s", got.skip)
			}
			if !overlap(got.before, tc.before) || !overlap(got.after, tc.after) || got.ratio > maxEditRatio {
				t.Errorf("got %q → %q (ratio %.3f), want around %q → %q", got.before, got.after, got.ratio, tc.before, tc.after)
			}
		})
	}
}

func TestDirectTerm(t *testing.T) {
	for _, tc := range []struct {
		name          string
		before, after string
		term, heard   string // "" : not term-like
	}{
		{name: "testDirectCandidateTermsFallbackReturnsFinalCorrectedToolName", before: "Cloud Code", after: "Claude Code", term: "Claude Code", heard: "Cloud Code"},
		{name: "punctuation at the edges goes", before: "Waxed.", after: "Voxt.", term: "Voxt", heard: "Waxed"},
		{name: "same word in another case", before: "voxt", after: "Voxt"},
		{name: "a word is deleted", before: "voxt", after: ""},
		{name: "a whole clause is not a term", before: "看一下我们投坑中有没有新的词源了。这个", after: "看一下我们投坑中有没有新的词元了。这个"},
		{name: "two edits far apart: more than four words", before: "one two three four five", after: "one two three four six"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			term, heard := directTerm(tc.before, tc.after)
			if term != tc.term || heard != tc.heard {
				t.Fatalf("got %q / %q, want %q / %q", term, heard, tc.term, tc.heard)
			}
		})
	}
}

// TestEditToSuggestion is the card's table: what a whole watch's edit
// proposes, with only local rules.
func TestEditToSuggestion(t *testing.T) {
	for _, tc := range []struct {
		name                    string
		inserted, baseline, end string
		term, heard             string // "": nothing proposed
	}{
		{name: "Cloud Code → Claude Code: only the changed word is the edit (1.x's window)",
			inserted: "please use Cloud Code today", baseline: "please use Cloud Code today", end: "please use Claude Code today",
			term: "Claude", heard: "Cloud"},
		{name: "cloud code → Claude Code: both words changed",
			inserted: "please use cloud code today", baseline: "please use cloud code today", end: "please use Claude Code today",
			term: "Claude Code", heard: "cloud code"},
		{name: "Go Host → Ghostty", inserted: "open Go Host now", baseline: "open Go Host now", end: "open Ghostty now", term: "Ghostty", heard: "Go Host"},
		{name: "Wechart → WeChat", inserted: "send on Wechart", baseline: "send on Wechart", end: "send on WeChat", term: "WeChat", heard: "Wechart"},
		{name: "cloud → Claude", inserted: "the cloud model", baseline: "the cloud model", end: "the Claude model", term: "Claude", heard: "cloud"},
		{name: "a Chinese word", inserted: "我在用 豆宝 写稿", baseline: "我在用 豆宝 写稿", end: "我在用 豆包 写稿", term: "豆包", heard: "豆宝"},
		{name: "a number changed", inserted: "see 3 notes", baseline: "see 3 notes", end: "see 4 notes"},
		{name: "a time changed", inserted: "meet at 10:30 ok", baseline: "meet at 10:30 ok", end: "meet at 11:30 ok"},
		{name: "am/pm kept, hour changed", inserted: "meet 3pm", baseline: "meet 3pm", end: "meet 4pm"},
		{name: "a Chinese time changed", inserted: "在下午三点开会", baseline: "在下午三点开会", end: "在下午四点开会"},
		{name: "only written on", inserted: "hello there", baseline: "hello there", end: "hello there and more"},
		{name: "outside the inserted text", inserted: "dictated", baseline: "typed\ndictated", end: "typed Edited\ndictated"},
		{name: "the whole sentence rewritten", inserted: "short note", baseline: "short note", end: "a completely different long paragraph about other things"},
		{name: "case only", inserted: "use voxt", baseline: "use voxt", end: "use Voxt"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got, why := suggestionFromEdit(tc.inserted, tc.baseline, tc.end)
			if tc.term == "" {
				if why == "" {
					t.Fatalf("proposed %+v", got)
				}
				return
			}
			if why != "" {
				t.Fatalf("not proposed: %s", why)
			}
			if got.Term != tc.term || !slices.Equal(got.Heard, []string{tc.heard}) || got.Source != FromEdit {
				t.Fatalf("got %+v, want %q heard %q", got, tc.term, tc.heard)
			}
			if a := got.AsTerm().Aliases; len(a) != 0 {
				t.Errorf("AsTerm has aliases %v", a)
			}
		})
	}
}

func TestIncompleteEdit(t *testing.T) {
	const baseline = "我们配合 Go Hoste 来实现 Terminal CLI 的输入。"
	for _, tc := range []struct {
		name     string
		baseline string
		end      string
		want     bool
	}{
		{"testContinuesObservingWhenLatestEditLooksLikeDeletionOnly", baseline, "我们配合  来实现 Terminal CLI 的输入。", true},
		{"testDoesNotContinueObservingWhenReplacementTextAlreadyExists", baseline, "我们配合 Ghostty 来实现 Terminal CLI 的输入。", false},
		{"testObservationSettlesAfterEchoedCommandContainsCompletedReplacement", "看一下我们投坑中有没有新的词源了。这个新的词源是也需要接飞的。", "看一下我们 token 中有没有新的词元了。这个新的词元也需要计费。", false},
		{"an unfinished deletion group beside a finished replacement", "特别是 Cloud Code 和我们的 Go Host。我们", "特别是 Claude Code 和我们的 。我们", true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got := incompleteEdit(tc.baseline, tc.end); got != tc.want {
				t.Fatalf("got %v", got)
			}
		})
	}
}
