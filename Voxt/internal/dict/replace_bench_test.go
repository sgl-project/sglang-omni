package dict

import (
	"fmt"
	"strings"
	"testing"

	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
)

func benchTerms(n int) []store.Term {
	ts := make([]store.Term, n)
	for i := range ts {
		ts[i] = store.Term{Term: fmt.Sprintf("Product%d", i), Aliases: []string{fmt.Sprintf("prod duct %d", i), fmt.Sprintf("产品%d号", i)}, Enabled: true}
	}
	return ts
}

func benchText(chars int) string {
	unit := "今天我们讨论一下 product 17 的发布计划，然后 prod duct 42 需要 review。The meeting covers roadmap items and budget. "
	var b strings.Builder
	for b.Len() < chars*2 {
		b.WriteString(unit)
	}
	return b.String()
}

func BenchmarkReplace(b *testing.B) {
	for _, c := range []struct{ terms, chars int }{{50, 300}, {200, 2000}, {1000, 2000}, {1000, 10000}} {
		ts, txt := benchTerms(c.terms), benchText(c.chars)
		b.Run(fmt.Sprintf("terms%d_bytes%d", c.terms, len(txt)), func(b *testing.B) {
			b.ReportAllocs()
			for b.Loop() {
				if Replace(txt, ts) == "" {
					b.Fatal("empty")
				}
			}
		})
	}
}
