package uitext

import "testing"

func TestMemoryBytes(t *testing.T) {
	for n, want := range map[int64]string{3_650_722_201: "3.4 GB", 1 << 30: "1.0 GB", 1_150 << 20: "1.1 GB", 512 << 20: "512 MB", 0: "0 MB"} {
		if got := MemoryBytes(n); got != want {
			t.Errorf("MemoryBytes(%d) = %q, want %q", n, got, want)
		}
	}
}
