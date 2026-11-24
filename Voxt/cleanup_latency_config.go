package main

import (
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/session"
)

const (
	cleanupFallbackEnvironment = "VOXT_EXPERIMENT_CLEANUP_FALLBACK_MS"
	maxCleanupFallbackMS       = int64(60_000)
)

func cleanupLatencyFromEnv() (session.CleanupLatencyPolicy, string) {
	raw, present := os.LookupEnv(cleanupFallbackEnvironment)
	if !present {
		return session.CleanupLatencyPolicy{}, ""
	}
	return parseCleanupLatency(raw)
}

func parseCleanupLatency(raw string) (session.CleanupLatencyPolicy, string) {
	milliseconds, err := strconv.ParseInt(strings.TrimSpace(raw), 10, 64)
	if err != nil || milliseconds < 0 || milliseconds > maxCleanupFallbackMS {
		return session.CleanupLatencyPolicy{}, cleanupFallbackEnvironment + " ignored; expected an integer from 0 to 60000 ms, experiment disabled"
	}
	if milliseconds == 0 {
		return session.CleanupLatencyPolicy{}, ""
	}
	return session.CleanupLatencyPolicy{AutoFallbackAfter: time.Duration(milliseconds) * time.Millisecond}, ""
}
