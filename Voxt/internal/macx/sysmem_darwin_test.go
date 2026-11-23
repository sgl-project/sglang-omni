//go:build darwin

package macx

import (
	"fmt"
	"os/exec"
	"strings"
	"testing"
)

// TestReadSystemMemory: the sysctls read as sysctl(8) prints them.
func TestReadSystemMemory(t *testing.T) {
	m, err := ReadSystemMemory()
	if err != nil {
		t.Fatal(err)
	}
	if m.Pressure != "normal" && m.Pressure != "warning" && m.Pressure != "critical" || m.AvailablePercent <= 0 || m.AvailablePercent > 100 {
		t.Errorf("%+v", m)
	}
	// "total = 16384.00M  used = 15579.75M  free = 804.25M  (encrypted)"
	out, err := exec.Command("sysctl", "-n", "vm.swapusage").Output()
	if err != nil {
		t.Fatal(err)
	}
	if f := strings.Fields(string(out)); len(f) < 3 || f[2] != fmt.Sprintf("%.2fM", float64(m.SwapTotalBytes)/(1<<20)) {
		t.Errorf("swap total %d bytes, sysctl says %q", m.SwapTotalBytes, out)
	}
	if m.SwapUsedBytes < 0 || m.SwapUsedBytes > m.SwapTotalBytes {
		t.Errorf("%+v", m)
	}
	t.Log(m, m.Swapins, m.Swapouts)
}
