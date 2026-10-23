package macx

import "fmt"

// SystemMemory is the Mac's memory state, for the log of a model that
// is slow: a model whose weights were swapped out slows down long before
// macOS calls the pressure critical.
type SystemMemory struct {
	// Pressure is macOS's memory-pressure level: "normal", "warning" or
	// "critical" (kern.memorystatus_vm_pressure_level).
	Pressure string
	// AvailablePercent is the share of memory macOS counts as available
	// (kern.memorystatus_level; memory_pressure's "free percentage").
	AvailablePercent int
	// SwapUsedBytes and SwapTotalBytes are the swap files' use (vm.swapusage).
	SwapUsedBytes, SwapTotalBytes int64
	// Swapins and Swapouts count the pages swapped in and out since boot
	// (vm_stat's Swapins and Swapouts); 0 when this macOS does not say.
	Swapins, Swapouts int64
}

func (m SystemMemory) String() string {
	return fmt.Sprintf("memory pressure %s, %d%% available, swap %.1f of %.1f GB used",
		m.Pressure, m.AvailablePercent, float64(m.SwapUsedBytes)/(1<<30), float64(m.SwapTotalBytes)/(1<<30))
}

// ReadSystemMemory reads the Mac's memory state (sysctl, a few
// microseconds).
func ReadSystemMemory() (SystemMemory, error) { return readSystemMemory() }
