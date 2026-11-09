package meeting

import (
	"context"
	"strings"
)

func (m *Manager) initTranslator() {
	if m.translator != nil {
		m.translator.Reset()
		return
	}
	if m.cfg.LLM == nil {
		return
	}
	m.translator = &LiveTranslator{
		OnDone: m.applyTranslation,
	}
	m.translator.Reset()
}

func (m *Manager) refreshTranslatorTarget() {
	if m.translator == nil {
		return
	}
	target := strings.TrimSpace(m.snap.Translate)
	if target == "" {
		m.translator.Reset()
		return
	}
	m.translator.Target = target
	if m.cfg.LLM != nil {
		chat, err := m.cfg.LLM(context.Background(), m.cfg.Settings().LLM)
		if err == nil {
			m.translator.Chat = chat
		}
	}
}

func (m *Manager) applyTranslation(segIdx int, text string) {
	ctx := context.Background()
	m.mu.Lock()
	seq := m.seq
	for i := range m.snap.Segments {
		if m.snap.Segments[i].StoreIdx == segIdx {
			m.snap.Segments[i].Translation = text
		}
	}
	s := m.snap
	m.mu.Unlock()
	m.notify(s)
	_ = m.cfg.Store.UpdateSegmentTranslation(ctx, seq, segIdx, text)
}
