package meeting

import (
	"context"
	"path/filepath"
	"runtime"
	"testing"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/store/sqlitestore"
)

func TestIsEcho(t *testing.T) {
	cases := []struct {
		name, mic, system string
		want              bool
	}{
		// #3982: the mic segment spans two system segments and hears one character more.
		{"heard from the speakers", "过的话，其实就是它会让你读一句话，然后一下出来，就是那整整齐齐的这个感觉。",
			"如果用过的话，其实就是它会让你读一句话，\n然后一下出来，就是那整整齐的这个感觉。", true},
		{"misheard words", "我想问一下，那个你说那个插件，它就是因为我其实对那个box的理解是",
			"我想问一下，那个你说那个插件，它就是因为我我我其实对那个Fox的理解是", true},
		{"English", "so the main window does not need to be fancy at all",
			"So the main window doesn't need to be fancy at all, right?", true},
		{"mixed", "我们用Voxt的translate功能试一下", "我们用 Voxt 的 translate 功能试一下吧", true},
		{"Chinese scripts", "我們今天討論語音識別和會議轉寫", "我们今天讨论语音识别和会议转写", true},
		{"own words over theirs", "因为就是我在想，我自己不是做app这个背景的，所以我觉得可能有点抽象",
			"嗯，是这样子，你们其实是坐在两个方向上，就是一个做的剪贴板", false},
		{"English own words", "I think we should ship the settings page first",
			"the results of the process are not what we expected", false},
		{"a short reply", "对。", "对，就是这样。", false},
		{"system silent", "我想问一下这个插件", "", false},
		{"empty mic", "", "我想问一下这个插件", false},
		{"mostly own words", "我先说一句我自己的看法，然后一下出来",
			"然后一下出来，就是那整整齐的这个感觉", false},
	}
	for _, c := range cases {
		if got := isEcho(c.mic, c.system); got != c.want {
			t.Errorf("%s: isEcho = %v, want %v", c.name, got, c.want)
		}
	}
}

func TestIsEchoChinesePhrase(t *testing.T) {
	if runtime.GOOS != "darwin" {
		t.Skip("CoreFoundation phrase rules run in Mac CI")
	}
	if !isEcho("我們看著會議轉寫", "我们看着会议转写") {
		t.Fatal("the same sentence in another script was not recognized")
	}
}

func TestDropEchoesLooksNearInTime(t *testing.T) {
	text := "然后一下出来，就是那整整齐齐的这个感觉。"
	sys := []Segment{{Source: SourceSystem, Start: 60 * time.Second, End: 68 * time.Second, Text: text}}
	mic := []Segment{
		{Source: SourceMic, Start: 61 * time.Second, End: 69 * time.Second, Text: text},
		{Source: SourceMic, Start: 69 * time.Second, End: 70 * time.Second, Text: text}, // within the slack
		{Source: SourceMic, Start: 10 * time.Second, End: 18 * time.Second, Text: text}, // the user said it a minute earlier
	}
	got := dropEchoes(mic, sys)
	if len(got) != 1 || got[0].Start != 10*time.Second {
		t.Fatalf("kept %+v", got)
	}
}

// The mic segment commits before the system track has committed the same
// speech: the system's live text still shows it is an echo.
func TestCommitSegmentDropsLiveEcho(t *testing.T) {
	ctx := context.Background()
	db, err := sqlitestore.Open(ctx, filepath.Join(t.TempDir(), "db"))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { db.Close() })
	m := NewManager(Config{Store: db.Meetings()})
	m.updateLive(SourceSystem, "我想问一下，那个你说那个插件，它就是因为我其实对那个Fox的理解是")
	m.commitSegment(Segment{Source: SourceMic, Start: 0, End: 8 * time.Second,
		Text: "我想问一下，那个你说那个插件，它就是因为我其实对那个box的理解是"})
	m.commitSegment(Segment{Source: SourceSystem, Start: 8 * time.Second, End: 16 * time.Second,
		Text: "然后一下出来，就是那整整齐齐的这个感觉。"})
	m.commitSegment(Segment{Source: SourceMic, Start: 9 * time.Second, End: 17 * time.Second,
		Text: "然后一下出来，就是那整整齐的这个感觉。"})
	m.commitSegment(Segment{Source: SourceMic, Start: 17 * time.Second, End: 20 * time.Second,
		Text: "我自己不是做app这个背景的，所以觉得有点抽象。"})
	m.segWrites.Wait()
	var got []string
	for _, s := range m.Current().Segments {
		got = append(got, string(s.Source)+": "+s.Text)
	}
	want := []string{"system: 然后一下出来，就是那整整齐齐的这个感觉。", "mic: 我自己不是做app这个背景的，所以觉得有点抽象。"}
	if len(got) != len(want) || got[0] != want[0] || got[1] != want[1] {
		t.Fatalf("segments %q, want %q", got, want)
	}
}
