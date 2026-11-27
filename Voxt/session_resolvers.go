package main

import (
	"context"
	"errors"
	"strings"

	"github.com/sgl-project/sglang-omni/Voxt/internal/local"
	"github.com/sgl-project/sglang-omni/Voxt/internal/session"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
)

// sessionResolvers opens speech and cleanup providers independently; the
// combined resolver remains for the other session paths.
func sessionResolvers(lm local.Services) (session.Resolver, session.ASRResolver, session.LLMResolver) {
	resolveASR := func(ctx context.Context, kind session.Kind, s settings.Settings) session.ASRReadiness {
		feature := featureOf(s, string(kind))
		id := strings.TrimSpace(feature.ASR)
		skip := false
		if id == "" {
			installed := lm.Installed(local.ASR)
			if model, ok := local.Choose(automaticSpeech(s, installed), ""); ok {
				id = model.ID
			} else {
				skip = len(installed) > 0
			}
		}
		result := session.ASRReadiness{}
		if !skip {
			provider, model, err := lm.ASR(ctx, id)
			if err != nil && !errors.Is(err, local.ErrNotInstalled) {
				return session.ASRReadiness{Err: err}
			}
			if err == nil {
				result.ASR, result.ASRInfo = provider, sessionModel(model)
			}
		}
		if voice, ok := result.ASR.(local.VoiceActivity); ok {
			result.VAD = voice.VAD()
		}
		return result
	}
	resolveLLM := func(ctx context.Context, kind session.Kind, s settings.Settings) session.LLMReadiness {
		if kind == store.KindTranscribe && !s.Features.Transcribe.Enhance {
			return session.LLMReadiness{Outcome: session.LLMNotRequested}
		}
		feature := featureOf(s, string(kind))
		ready := local.ForFeature(lm.Installed(local.LLM), string(kind))
		want := strings.TrimSpace(feature.LLM)
		model, ok := local.Choose(ready, want)
		if !ok {
			return session.LLMReadiness{Outcome: session.LLMUnavailable}
		}
		if want != "" && model.ID != want {
			model = local.Model{}
		}
		result := session.LLMReadiness{Outcome: session.LLMUnavailable}
		if model.ID != "" {
			client, opened, err := lm.LLM(ctx, model.ID)
			if err != nil {
				result.Err = err
			} else if opened.ID == model.ID {
				result.LLM, result.LLMInfo = client, sessionModel(opened)
			}
		}
		for _, id := range fallbackCandidates(kind, s, ready, model.ID) {
			client, opened, err := lm.LLM(ctx, id)
			if err != nil || opened.ID != id {
				if err != nil {
					result.Err = err
				}
				continue
			}
			if result.LLM == nil {
				result.LLM, result.LLMInfo = client, sessionModel(opened)
			} else {
				result.Fallback, result.FallbackInfo = client, sessionModel(opened)
			}
			break
		}
		if result.LLM != nil {
			result.Outcome = session.LLMAvailable
		}
		return result
	}
	resolve := func(ctx context.Context, kind session.Kind, s settings.Settings, speech bool) (session.Services, error) {
		var services session.Services
		if speech {
			resolved := resolveASR(ctx, kind, s)
			if resolved.Err != nil {
				return session.Services{}, resolved.Err
			}
			services.ASR, services.ASRInfo, services.VAD = resolved.ASR, resolved.ASRInfo, resolved.VAD
		}
		resolved := resolveLLM(ctx, kind, s)
		services.LLM, services.LLMInfo = resolved.LLM, resolved.LLMInfo
		services.Fallback, services.FallbackInfo = resolved.Fallback, resolved.FallbackInfo
		return services, nil
	}
	return resolve, resolveASR, resolveLLM
}

// resolver is the compatibility adapter for retries and non-voice paths.
func resolver(lm local.Services) session.Resolver {
	resolve, _, _ := sessionResolvers(lm)
	return resolve
}

func sessionModel(model local.Model) session.Model {
	return session.Model{ID: model.ID, Name: model.Name, Model: model.ID, Remote: model.Remote}
}
