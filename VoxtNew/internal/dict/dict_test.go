package dict

import (
	"errors"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/sgl-project/sglang-omni/VoxtNew/internal/store"
)

var t0 = time.Date(2026, 10, 15, 9, 0, 0, 0, time.UTC)

func TestCleanAndCheck(t *testing.T) {
	got := Clean(store.Term{Term: "  Vox\ntunew ", Aliases: []string{" vox tunew", "VOX TUNEW", "", "Vox tunew", "voxtnew", "vox  tuna "}, Note: " n "})
	want := store.Term{Term: "Vox tunew", Aliases: []string{"voxtnew", "vox tuna"}, Note: "n"}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("Clean = %+v", got)
	}
	many := store.Term{Term: "x"}
	for i := range 20 {
		many.Aliases = append(many.Aliases, strings.Repeat("a", i+1))
	}
	if n := len(Clean(many).Aliases); n != MaxAliases {
		t.Errorf("%d aliases", n)
	}
	for in, want := range map[string]error{"VoxtNew": nil, "": ErrEmpty, "  ": ErrEmpty, "a\nb": ErrNewline, strings.Repeat("长", 81): ErrLong, strings.Repeat("长", 80): nil} {
		if err := Check(store.Term{Term: in}); !errors.Is(err, want) && err != want {
			t.Errorf("Check(%q) = %v", in, err)
		}
	}
	if got := SplitAliases("vox tunew, 沃克斯牛，沃克斯 、 x; y\n z,,"); !reflect.DeepEqual(got, []string{"vox tunew", "沃克斯牛", "沃克斯", "x", "y", "z"}) {
		t.Errorf("SplitAliases = %q", got)
	}
}

func TestHints(t *testing.T) {
	terms := []store.Term{
		{Term: "VoxtNew", Enabled: true, UpdatedAt: t0},
		{Term: "Kubernetes", Enabled: true, UpdatedAt: t0.Add(2 * time.Hour)},
		{Term: "Off", Enabled: false, UpdatedAt: t0.Add(3 * time.Hour)},
		{Term: "voxtnew", Enabled: true, UpdatedAt: t0.Add(-time.Hour)},
		{Term: "  火山 引擎 ", Aliases: []string{"火山引擎"}, Enabled: true, UpdatedAt: t0.Add(time.Hour)},
		{Term: " ", Enabled: true},
	}
	if got := Hints(terms); !reflect.DeepEqual(got, []string{"Kubernetes", "火山 引擎", "VoxtNew"}) {
		t.Errorf("Hints = %q", got)
	}
	if Hints(nil) != nil {
		t.Error("hints of nothing")
	}
}

func TestExportImport(t *testing.T) {
	terms := []store.Term{
		{Term: "VoxtNew", Aliases: []string{"vox tunew", "沃克斯牛"}, Note: "our app, \"quoted\", with a comma", Enabled: true},
		{Term: "MyGo", Enabled: false},
	}
	csvData := Export(terms, CSV)
	if !strings.HasPrefix(string(csvData), "\ufeffterm,aliases,note,enabled\n") {
		t.Errorf("csv %q", csvData)
	}
	back, skipped, err := Import(csvData, CSV)
	if err != nil || skipped != 0 || !reflect.DeepEqual(back, terms) {
		t.Errorf("csv round trip %+v %d %v", back, skipped, err)
	}
	txt := Export(terms, Text)
	if !strings.Contains(string(txt), "\nVoxtNew = vox tunew, 沃克斯牛\n# MyGo\n") {
		t.Errorf("txt %q", txt)
	}
	back, _, err = Import(txt, Text)
	if err != nil || len(back) != 1 || back[0].Term != "VoxtNew" || len(back[0].Aliases) != 2 {
		t.Errorf("txt round trip %+v %v", back, err)
	}
}

func TestImportVariants(t *testing.T) {
	for _, tc := range []struct {
		name, format, data string
		want               []store.Term
		skipped            int
	}{
		{"csv without a header", CSV, "VoxtNew,vox tunew;voxtuna,note\n,,\nMyGo\n", []store.Term{
			{Term: "VoxtNew", Aliases: []string{"vox tunew", "voxtuna"}, Note: "note", Enabled: true},
			{Term: "MyGo", Enabled: true},
		}, 1},
		{"csv header in another order, Chinese names", CSV, "启用,备注,词条,别名\n否,内部,豆包,\"dou bao, 逗包\"\n", []store.Term{
			{Term: "豆包", Aliases: []string{"dou bao", "逗包"}, Note: "内部", Enabled: false},
		}, 0},
		{"csv with an unknown column", CSV, "Term,Weight\nVoxtNew,5\n", []store.Term{{Term: "VoxtNew", Enabled: true}}, 0},
		{"text with tabs and comments", Text, "# list\n\nKubernetes\tk8s, kube\nDeepSeek = deep seek\n// skip\r\nGo\n", []store.Term{
			{Term: "Kubernetes", Aliases: []string{"k8s", "kube"}, Enabled: true},
			{Term: "DeepSeek", Aliases: []string{"deep seek"}, Enabled: true},
			{Term: "Go", Enabled: true},
		}, 0},
		{"text with a long line", Text, strings.Repeat("x", 81) + "\nok\n", []store.Term{{Term: "ok", Enabled: true}}, 1},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got, skipped, err := Import([]byte(tc.data), tc.format)
			if err != nil || skipped != tc.skipped || !reflect.DeepEqual(got, tc.want) {
				t.Errorf("Import = %+v, %d, %v", got, skipped, err)
			}
		})
	}
	if _, _, err := Import([]byte{0xff, 0xfe, 0x00}, CSV); !errors.Is(err, ErrFormat) {
		t.Errorf("binary: %v", err)
	}
	if FormatOf("terms.TXT") != Text || FormatOf("terms.csv") != CSV || FormatOf("x") != CSV {
		t.Error("FormatOf")
	}
}

func TestMerge(t *testing.T) {
	have := []store.Term{
		{ID: "1", Term: "VoxtNew", Aliases: []string{"vox tunew"}, Enabled: true, CreatedAt: t0},
		{ID: "2", Term: "MyGo", Note: "kept", Enabled: false},
		{ID: "3", Term: "Same", Aliases: []string{"x"}, Enabled: true},
	}
	imported := []store.Term{
		{Term: "voxtnew", Aliases: []string{"Vox Tunew", "沃克斯牛"}, Note: "new note", Enabled: true},
		{Term: "MyGo", Note: "ignored", Enabled: true},
		{Term: "Same", Aliases: []string{"X"}},
		{Term: "New", Enabled: true},
		{Term: "new", Aliases: []string{"neu"}},
	}
	put, added, updated := Merge(have, imported)
	if added != 1 || updated != 1 {
		t.Errorf("added %d updated %d", added, updated)
	}
	want := []store.Term{
		{ID: "1", Term: "VoxtNew", Aliases: []string{"vox tunew", "沃克斯牛"}, Note: "new note", Enabled: true, CreatedAt: t0},
		{Term: "New", Aliases: []string{"neu"}, Enabled: true},
	}
	if !reflect.DeepEqual(put, want) {
		t.Errorf("put %+v", put)
	}
}
