package omni

import (
	"runtime"
	"testing"
	"time"
)

func BenchmarkFinalMultipart(b *testing.B) {
	for _, duration := range []time.Duration{60 * time.Second, 120 * time.Second} {
		b.Run(duration.String(), func(b *testing.B) {
			request := finalRequest{pcm: tone(duration, 0.3), language: "en", context: "SGLang", maxTokens: 1024}
			b.ReportAllocs()
			for b.Loop() {
				newBody, _, _ := request.body("qwen3-asr")
				runtime.KeepAlive(newBody())
			}
		})
	}
}
