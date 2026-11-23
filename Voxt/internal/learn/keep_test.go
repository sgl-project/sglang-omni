package learn

import "testing"

func TestKeep(t *testing.T) {
	travel := "北京到大同今天的航班和车次有哪些？有没有 K130 航班？"
	for _, c := range []struct {
		term, record string
		want         bool
	}{
		// The terms a scan should keep.
		{"Ghostty", "I tried Ghostty today", true},
		{"豆包", "把豆包接进来", true},
		{"OpenAI", "我今天要给 OpenAI 的接口做联调。", true},
		{"MCP", "the MCP server", true},
		{"旧金山", "下周去旧金山", true},
		// Found anywhere in the records, ignoring case.
		{"ghostty", "I tried GHOSTTY today", true},
		// The same word off a trip is a name.
		{"北京", "北京的同事说", true},
		// Generic English and Chinese words, any case.
		{"email", "send an email", false},
		{"Company", "our Company", false},
		{"token", "a token", false},
		{"会议", "下午的会议", false},
		{"航班", "这个航班", false},
		{"车次", "哪个车次", false},
		// "my/our/this … generic noun".
		{"our model", "our model is good", false},
		{"this issue", "this issue again", false},
		// Chinese reference phrases.
		{"我们的规则", "按我们的规则", false},
		{"这个功能", "这个功能不错", false},
		{"这个苹果", "这个苹果很好吃", true},
		// Structure: lengths, punctuation, digits, newlines.
		{"   ", "anything at all", false},
		{"this is a very long generic transcript phrase", "this is a very long generic transcript phrase", false},
		{"今天我们要开会讨论一下", "今天我们要开会讨论一下", false},
		{"wrong term, maybe", "wrong term, maybe", false},
		{"20260413", "20260413", false},
		{"一二", "一二", true},
		{"123", "room 123", false},
		{"abcdefghijklmnopqrstuvwxyzabcdefghijklmn", "abcdefghijklmnopqrstuvwxyzabcdefghijklmn", false},
		{"one two three four five six seven", "one two three four five six seven", false},
		{"a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1", "a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1", false},
		{"two\nlines", "two\nlines", false},
		// URLs and paths.
		{"https://voxt.dev", "go to https://voxt.dev now", false},
		{"voxt.dev", "go to voxt.dev now", false},
		{"/usr/local/bin", "put it in /usr/local/bin", false},
		{"/opt", "put it in /opt", false},
		{"~/Documents", "in ~/Documents", false},
		{"src/learn/keep", "see src/learn/keep today", false},
		{`docs\notes`, `open docs\notes`, false},
		{"CI/CD", "our CI/CD pipeline", true},
		// Only what is in the records.
		{"Invented", "nothing like it here", false},
		// Context: a route's ends and a transport number, on a trip only.
		{"K130", travel, false},
		{"北京", travel, false},
		{"大同", travel, false},
		{"MU5735", "我坐 MU5735 这趟航班", false},
		{"G1234", "G1234 高铁到了", false},
		{"K130", "the K130 keyboard is great", true},
		{"K130", "the K130 keyboard: this sentence runs on and on until the flight", true},
		{"Boston", "flights from Boston", false},
		{"Boston", "Boston to Denver train", false},
		{"Denver", "a train to Denver", false},
		{"Kubernetes", "the train of thought on Kubernetes", true},
	} {
		if got := keep(c.term, []string{c.record}); got != c.want {
			t.Errorf("keep(%q) in %q = %v, want %v", c.term, c.record, got, c.want)
		}
	}
}

func TestKeepFindsTheTermInAnyRecord(t *testing.T) {
	if !keep("Ghostty", []string{"nothing here", "I tried Ghostty"}) {
		t.Error("the second record was not read")
	}
	if keep("大同", []string{"北京到大同的航班", "大同的同事来了"}) {
		t.Error("the first record that has the term decides, as in 1.x")
	}
}
