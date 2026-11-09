package omni

import (
	"slices"

	"github.com/sgl-project/sglang-omni/Voxt/internal/asr"
)

// previewBatch is the live text of a model without a realtime API (Voxt
// 1.x's batch preview): once ASRTuning.DecodeInterval of new audio came,
// a Final request over the audio no piece holds yet, its text after the
// pieces'. The requests take turns with the pieces' on the one server, so
// a slow model previews less often. A request that fails stops the
// preview; the Final still runs at Finish.
func (s *live) previewBatch() {
	t := s.a.spec.ASR
	step := samplesOf(t.DecodeInterval) * 2
	shown := 0 // the audio's bytes at the last preview
	for {
		select {
		case <-s.tick:
		case <-s.ctx.Done():
			return
		}
		s.mu.Lock()
		finished, audio, language, done := s.finished, s.audio, s.language, 0
		var texts []string
		if pc := s.pc; pc != nil {
			done, texts = pc.done, slices.Clone(pc.texts)
			if pc.language != "" {
				language = pc.language
			}
		}
		s.mu.Unlock()
		if finished {
			return
		}
		if len(audio)-shown < step || len(audio)-done < step {
			continue
		}
		shown = len(audio)
		tail := audio[done:]
		req := finalRequest{pcm: padPiece(tail, t), language: language, context: s.bias.context,
			maxTokens: t.tokenBudget(asr.Duration(tail, sampleRate)), layout: t.AudioLayout, preview: true}
		r, err := transcribe(s.ctx, s.l, req, s.a.rt.cfg.StallTimeout)
		if err != nil {
			if s.ctx.Err() == nil {
				s.a.rt.logf("warning", "%s: live preview stopped: %v", s.a.spec.ID, err)
			}
			return
		}
		s.mu.Lock()
		if !s.finished {
			if text, ok := s.asm.apply(0, 0, joinParts(append(texts, r.text)), false); ok {
				s.previewed = shown
				s.emit(asr.Event{Kind: asr.EventPartial, Text: text})
			}
		}
		s.mu.Unlock()
	}
}
