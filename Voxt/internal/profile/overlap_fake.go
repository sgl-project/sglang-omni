package profile

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"time"
)

const (
	simulatedCleanup       = "cleanup"
	simulatedSummaryMap    = "summary_map"
	simulatedSummaryReduce = "summary_reduce"
)

type simulatedLLMRequest struct {
	Operation       string          `json:"operation"`
	Parts           []SimulatedPart `json:"parts"`
	PreviousContext string          `json:"previous_context,omitempty"`
}

type overlapLLM interface {
	Call(context.Context, simulatedLLMRequest) (string, error)
}

type simulatedLLM struct {
	latency time.Duration
	sleep   func(context.Context, time.Duration) error
}

func newSimulatedLLM(latency time.Duration) *simulatedLLM {
	return &simulatedLLM{latency: latency, sleep: sleepContext}
}

func sleepContext(ctx context.Context, duration time.Duration) error {
	if duration <= 0 {
		return ctx.Err()
	}
	timer := time.NewTimer(duration)
	defer timer.Stop()
	select {
	case <-timer.C:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}

func (s *simulatedLLM) Call(ctx context.Context, request simulatedLLMRequest) (string, error) {
	if err := ctx.Err(); err != nil {
		return "", err
	}
	if s.latency > 0 {
		if err := s.sleep(ctx, s.latency); err != nil {
			return "", err
		}
	}
	switch request.Operation {
	case simulatedCleanup:
		parts := make([]SimulatedPart, len(request.Parts))
		for index, part := range request.Parts {
			parts[index] = SimulatedPart{Index: part.Index, Text: strings.TrimSpace(part.Text)}
		}
		encoded, err := json.Marshal(parts)
		return string(encoded), err
	case simulatedSummaryMap, simulatedSummaryReduce:
		input, err := json.Marshal(request)
		if err != nil {
			return "", err
		}
		response := struct {
			Summary struct {
				Title   string `json:"title"`
				Content string `json:"content"`
			} `json:"transcript_summary"`
			Todos []string `json:"todo_list"`
		}{Todos: []string{}}
		response.Summary.Title = "simulated " + strings.TrimPrefix(request.Operation, "summary_")
		response.Summary.Content = request.Operation + ":" + HashText(string(input))
		encoded, err := json.Marshal(response)
		return string(encoded), err
	default:
		return "", fmt.Errorf("overlap: unknown simulated operation %q", request.Operation)
	}
}
