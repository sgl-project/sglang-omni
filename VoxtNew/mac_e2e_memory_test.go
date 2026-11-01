package main

// The loaded models' memory on a Mac (VOXTNEW_MAC_E2E=1, the runtimes in
// VOXTNEW_OMNI_RUNTIME and VOXTNEW_LLM, the models in the model library,
// $HF_HUB_CACHE or ~/.cache/huggingface/hub): the app's own local models load asr.fast,
// then llm.fast and llm.translate, and the Local models page's
// measurement (macx.Memory by pid) is compared with what voxtnew-llm
// says of itself. It logs MEASURE lines:
//
//	MAC_DIR=code/VoxtNew-xyz scripts/mac.sh 'VOXTNEW_MAC_E2E=1 \
//	  VOXTNEW_OMNI_RUNTIME=$HOME/voxtnew-runtime/omni VOXTNEW_LLM=$HOME/voxtnew-runtime/llm/voxtnew-llm \
//	  go test -count=1 -run TestMacModelMemory -v .'

import (
	"context"
	"os"
	"runtime"
	"testing"
	"time"

	"github.com/sgl-project/sglang-omni/VoxtNew/internal/llm"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/macx"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/models"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/settings"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/uitext"
)

func TestMacModelMemory(t *testing.T) {
	if os.Getenv("VOXTNEW_MAC_E2E") != "1" || runtime.GOOS != "darwin" {
		t.Skip("set VOXTNEW_MAC_E2E=1 on a Mac, with VOXTNEW_OMNI_RUNTIME and VOXTNEW_LLM")
	}
	for _, id := range []string{models.ASRFast, models.VADSilero, models.LLMFast, models.LLMTranslate} {
		if m, _ := models.Lookup(id); models.Installed(realHub, m) == "" {
			t.Fatalf("%s is not in the model library %s", id, realHub)
		}
	}
	svc := realLocal{hub: realHub}
	s := newSessionApp(t, "", func(*settings.Settings) {}, svc)
	a := s.app
	ctx := context.Background()

	if _, _, err := a.local.ASR(ctx, models.ASRFast); err != nil {
		t.Fatal(err)
	}
	var pid int
	for end := time.Now().Add(3 * time.Minute); pid == 0; time.Sleep(100 * time.Millisecond) {
		if time.Now().After(end) {
			t.Fatal("the speech server did not start")
		}
		for _, p := range a.omniRT.Processes() {
			if p.ID == models.ASRFast {
				pid = p.PID
			}
		}
	}
	m, err := macx.Memory(pid)
	t.Logf("MEASURE %s pid %d: footprint %s, resident %s (%v)", models.ASRFast, pid, mb(m.Footprint), mb(m.Resident), err)
	if err != nil || m.InUse() < 100<<20 {
		t.Errorf("%s: %+v %v", models.ASRFast, m, err)
	}

	for _, id := range []string{models.LLMFast, models.LLMTranslate} {
		chat, _, err := a.local.LLM(ctx, id)
		if err != nil {
			t.Fatal(err)
		}
		req := llm.Request{Messages: []llm.Message{{Role: llm.User, Content: "Say hello."}}, Skill: llm.SkillRewrite}
		if id == models.LLMTranslate {
			req = llm.Request{Skill: llm.SkillTranslateText, Translate: &llm.TranslateHint{Text: "你好", Target: "en"}}
		}
		if _, err := chat.Chat(ctx, req); err != nil {
			t.Fatal(err)
		}
		st := a.llmMade().Status()
		m, err := macx.Memory(st.PID)
		t.Logf("MEASURE %s pid %d: footprint %s, resident %s (%v); it reported footprint %s, resident %s",
			id, st.PID, mb(m.Footprint), mb(m.Resident), err, mb(st.Memory.FootprintBytes), mb(st.Memory.ResidentBytes))
		reported := max(st.Memory.FootprintBytes, st.Memory.ResidentBytes)
		if err != nil || m.InUse() < reported/2 || m.InUse() > reported*2 {
			t.Errorf("%s: measured %+v, reported %+v (%v)", id, m, st.Memory, err)
		}
	}

	// What the page shows.
	a.readModelMemory()
	s.pump("the measure", func() bool { return len(a.modelMem) >= 3 })
	for id, b := range a.modelMem {
		t.Logf("MEASURE page %s: %s", id, modelMemText(b))
	}
	if a.modelMem[models.ASRFast] <= 0 || a.modelMem[models.LLMTranslate] <= 0 {
		t.Errorf("page %v", a.modelMem)
	}
}

func mb(b int64) string { return uitext.MemoryBytes(b) }
