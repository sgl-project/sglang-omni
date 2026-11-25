package session

import "testing"

// TestHeadMatch: an answer to the live text serves the final text that
// says the same, with the words they differ in put right; anything that
// may change more than the words is asked again.
func TestHeadMatch(t *testing.T) {
	for _, tc := range []struct {
		name, live, final, answer, want string // want "": asked again
	}{
		{"punctuation", "我们周三开会 地点在二楼。", "我们周三开会，地点在二楼。", "我们周三开会，地点在二楼。", "我们周三开会，地点在二楼。"},
		{"a word", "得以在圣彼得广场举行的葬礼，出席者太多了。", "对于在圣彼得广场举行的葬礼，出席者太多了。", "得以在圣彼得广场举行的葬礼，出席者太多了。", "对于在圣彼得广场举行的葬礼，出席者太多了。"},
		{"an English word", "I like the read car.", "I like the red car.", "I like the read car.", "I like the red car."},
		{"a word added", "We ship it Friday.", "We will ship it Friday.", "We ship it Friday.", "We will ship it Friday."},
		{"a word left out", "这个方案那么还要再改。", "这个方案还要再改。", "这个方案那么还要再改。", "这个方案还要再改。"},
		{"fillers only", "那我们就这样。", "呃那我们就这样。", "那我们就这样。", "那我们就这样。"},
		{"a number written otherwise", "大概没有达到6.5级。", "大概没有达到六点五级。", "大概没有达到6.5级。", "大概没有达到6.5级。"},
		{"another number", "一共五十八块。", "一共六十八块。", "一共58块。", ""},
		{"a correction", "我们周二开会。", "我们周二不对周三开会。", "我们周二开会。", ""},
		{"a question", "你来。", "你来？", "你来。", ""},
		{"many differences", "今天天气真好我们出去玩吧。", "昨天雨很大大家都没有出门啊。", "今天天气真好，我们出去玩。", ""},
		{"many words more", "We ship on Friday.", "We ship the new build with the fixes for the two crash reports on Friday.", "We ship on Friday.", ""},
		{"many words less", "We ship the new build with the fixes on Friday.", "We ship on Friday.", "We ship the new build with the fixes on Friday.", ""},
		{"three words", "这个方案很好。", "这个方案真的很好。", "这个方案很好。", "这个方案真的很好。"},
		{"not in the answer", "I like the read car.", "I like the red car.", "The auto is nice.", ""},
		{"twice in the answer", "在广场在广场。", "在广场在河边。", "在广场在广场。", "在广场在河边。"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			old := spokenWords(tc.live)
			got := ""
			if changes, ok := headChanges(old, tc.live, tc.final); ok {
				if text, ok := putRight(tc.answer, old, changes); ok {
					got = text
				}
			}
			if got != tc.want {
				t.Errorf("got %q, want %q", got, tc.want)
			}
		})
	}
}

// TestTailAfter: the live text's rest after the finalised stretches is
// found where they end, though the two differ, and not when the live text
// does not reach there or it would be in two places.
func TestTailAfter(t *testing.T) {
	final := "我们先把合同发给法务的王老师审一下，然后再安排时间。"
	for _, tc := range []struct{ name, live, want string }{
		{"the same", final + "下周三下午开会", "下周三下午开会"},
		{"other words before", "我们先把合同发给法务的汪老师审一下，然后再安排时间，下周三下午开会", "下周三下午开会"},
		{"not reached", "我们先把合同发给法务", ""},
		{"nothing after", final, ""},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got, ok := tailAfter(tc.live, final)
			if got != tc.want || ok != (tc.name != "not reached") {
				t.Errorf("got %q, %v", got, ok)
			}
		})
	}
	if got := joinSpeech("We met.", "Then we left"); got != "We met. Then we left" {
		t.Errorf("joinSpeech %q", got)
	}
	if got := joinSpeech("我们开会。", "然后 go"); got != "我们开会。然后 go" {
		t.Errorf("joinSpeech %q", got)
	}
}
