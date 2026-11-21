package profile

import (
	"errors"
	"os"
	"testing"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/macx"
)

func TestPressureRankTreatsUnknownLevelsAsCritical(t *testing.T) {
	if got := pressureRank("level 2"); got != pressureRank("critical") {
		t.Fatalf("unknown pressure rank = %d, want critical", got)
	}
}

func TestResourceSamplerPeakCPUAndSwap(t *testing.T) {
	sampler := NewResourceSampler(func() []int { return []int{10, 11, 11} })
	memoryCalls := 0
	sampler.memory = func(pid int) (macx.ProcMemory, error) {
		memoryCalls++
		extra := int64(0)
		if memoryCalls > 2 {
			extra = 100
		}
		return macx.ProcMemory{Resident: int64(pid) + extra, Footprint: int64(pid*2) + extra}, nil
	}
	sampler.cpu = func(pid int) (macx.ProcCPU, error) {
		return macx.ProcCPU{User: time.Duration(pid) * time.Millisecond, System: time.Millisecond}, nil
	}
	sampler.Sample()
	sampler.Sample()
	start := MemorySample{Swapins: Available(int64(8)), Swapouts: Available(int64(3))}
	end := MemorySample{Swapins: Available(int64(15)), Swapouts: Available(int64(9))}
	report := sampler.report(start, end, HostSample{}, HostSample{})
	if report.PeakRSSBytes.Value == nil || *report.PeakRSSBytes.Value != 221 {
		t.Errorf("peak RSS %+v", report.PeakRSSBytes)
	}
	if report.PeakPhysicalFootprint.Value == nil || *report.PeakPhysicalFootprint.Value != 242 {
		t.Errorf("peak footprint %+v", report.PeakPhysicalFootprint)
	}
	if report.CPUUserMS.Value == nil || *report.CPUUserMS.Value != 21 || report.CPUSystemMS.Value == nil || *report.CPUSystemMS.Value != 2 {
		t.Errorf("CPU %+v", report)
	}
	if len(report.ProcessPeaks) != 2 || report.ProcessPeaks[0].RSSBytes != 110 || report.ProcessPeaks[0].FootprintBytes != 120 || report.ProcessPeaks[1].RSSBytes != 111 || report.ProcessPeaks[1].FootprintBytes != 122 {
		t.Errorf("per-process peaks %+v", report.ProcessPeaks)
	}
	if report.SwapinsDelta.Value == nil || *report.SwapinsDelta.Value != 7 || report.SwapoutsDelta.Value == nil || *report.SwapoutsDelta.Value != 6 {
		t.Errorf("swap %+v", report)
	}
	if report.GPUMemoryBytes.Value != nil || report.GPUTimeMS.Value != nil || report.EnergyJoules.Value != nil {
		t.Error("unavailable GPU/energy metrics acquired values")
	}
}

func TestResourceSamplerRejectsPartialPeak(t *testing.T) {
	sampler := NewResourceSampler(func() []int { return []int{10, 11} })
	sampler.memory = func(pid int) (macx.ProcMemory, error) {
		if pid == 11 {
			return macx.ProcMemory{}, errors.New("native process exited")
		}
		return macx.ProcMemory{Resident: 100, Footprint: 200}, nil
	}
	sampler.Sample()
	report := sampler.report(MemorySample{}, MemorySample{}, HostSample{}, HostSample{})
	if report.PeakRSSBytes.Value != nil || report.PeakPhysicalFootprint.Value != nil {
		t.Fatalf("a harness-only sample was reported as a complete peak: %+v", report)
	}
}

func TestResourceSamplerUnavailableIsNull(t *testing.T) {
	sampler := NewResourceSampler(func() []int { return []int{10} })
	sampler.memory = func(int) (macx.ProcMemory, error) { return macx.ProcMemory{}, errors.New("memory denied") }
	sampler.cpu = func(int) (macx.ProcCPU, error) { return macx.ProcCPU{}, errors.New("CPU denied") }
	sampler.Sample()
	report := sampler.report(MemorySample{}, MemorySample{}, HostSample{}, HostSample{})
	if report.PeakRSSBytes.Value != nil || report.PeakRSSBytes.UnavailableReason != "memory denied" {
		t.Errorf("%+v", report.PeakRSSBytes)
	}
	if report.CPUUserMS.Value != nil || report.CPUUserMS.UnavailableReason != "CPU denied" {
		t.Errorf("%+v", report.CPUUserMS)
	}
	if report.SwapinsDelta.Value != nil || report.SwapinsDelta.UnavailableReason == "" {
		t.Errorf("%+v", report.SwapinsDelta)
	}
}

func TestResourceSamplerCapturesTransientMemoryPressureWarning(t *testing.T) {
	sampler := NewResourceSampler(func() []int { return []int{10} })
	sampler.memory = func(int) (macx.ProcMemory, error) { return macx.ProcMemory{Resident: 100, Footprint: 200}, nil }
	sampler.cpu = func(int) (macx.ProcCPU, error) { return macx.ProcCPU{}, nil }
	pressures := []macx.SystemMemory{
		{Pressure: "normal", AvailablePercent: 70, AvailablePercentValid: true, SwapUsedBytes: 100, SwapValid: true},
		{Pressure: "warning", AvailablePercent: 35, AvailablePercentValid: true, SwapUsedBytes: 500, SwapValid: true},
		{Pressure: "normal", AvailablePercent: 65, AvailablePercentValid: true, SwapUsedBytes: 500, SwapValid: true},
	}
	i := 0
	sampler.sysmem = func() (macx.SystemMemory, error) {
		m := pressures[min(i, len(pressures)-1)]
		i++
		return m, nil
	}
	for range pressures {
		sampler.Sample()
	}
	report := sampler.report(MemorySample{Pressure: Available("normal")}, MemorySample{Pressure: Available("normal")}, HostSample{}, HostSample{})
	if report.PeakMemoryPressure.Value == nil || *report.PeakMemoryPressure.Value != "warning" {
		t.Fatalf("peak_memory_pressure = %+v, want warning", report.PeakMemoryPressure)
	}
	if len(report.MemoryPressureWarnings) != 1 || report.MemoryPressureWarnings[0].Pressure != "warning" {
		t.Fatalf("memory_pressure_warnings = %+v, want 1 warning event", report.MemoryPressureWarnings)
	}
	if report.PeakSwapUsedBytes.Value == nil || *report.PeakSwapUsedBytes.Value != 500 {
		t.Fatalf("peak_swap_used_bytes = %+v, want 500", report.PeakSwapUsedBytes)
	}
}

func TestResourceSamplerSnapshotSplitsProcessesAndKeepsPressure(t *testing.T) {
	self := os.Getpid()
	native := self + 1
	sampler := NewResourceSampler(func() []int { return []int{self, native, native} })
	sampler.memory = func(pid int) (macx.ProcMemory, error) {
		return macx.ProcMemory{Resident: int64(pid), Footprint: int64(pid * 10)}, nil
	}
	sampler.sysmem = func() (macx.SystemMemory, error) {
		return macx.SystemMemory{Pressure: "critical", AvailablePercent: 12, AvailablePercentValid: true}, nil
	}
	sample := sampler.Snapshot("ready")
	if sample.Label != "ready" || sample.Pressure != "critical" {
		t.Fatalf("snapshot %+v", sample)
	}
	if sample.AvailablePercent == nil || *sample.AvailablePercent != 12 {
		t.Fatalf("available percent %+v", sample.AvailablePercent)
	}
	if len(sample.Processes) != 2 {
		t.Fatalf("processes %+v", sample.Processes)
	}
	roles := map[int]ProcessPhaseMemory{}
	for _, process := range sample.Processes {
		roles[process.PID] = process
	}
	if roles[self].Role != "harness" || roles[native].Role != "native" {
		t.Fatalf("roles %+v", sample.Processes)
	}
	if roles[native].FootprintBytes != int64(native*10) || roles[self].RSSBytes != int64(self) {
		t.Fatalf("per-process memory %+v", sample.Processes)
	}
	report := sampler.report(MemorySample{}, MemorySample{}, HostSample{}, HostSample{})
	if len(report.PhaseSamples) != 1 || report.PhaseSamples[0].Label != "ready" {
		t.Fatalf("report phase samples %+v", report.PhaseSamples)
	}
}

func TestResourceSamplerSnapshotToleratesProcessMemoryErrors(t *testing.T) {
	sampler := NewResourceSampler(func() []int { return []int{10, 11} })
	sampler.memory = func(pid int) (macx.ProcMemory, error) {
		if pid == 11 {
			return macx.ProcMemory{}, errors.New("native process exited")
		}
		return macx.ProcMemory{Resident: 100, Footprint: 200}, nil
	}
	sampler.sysmem = func() (macx.SystemMemory, error) { return macx.SystemMemory{}, errors.New("denied") }
	sample := sampler.Snapshot("pre_load")
	if len(sample.Processes) != 1 || sample.Processes[0].PID != 10 {
		t.Fatalf("processes %+v", sample.Processes)
	}
	if sample.Pressure != "" || sample.AvailablePercent != nil {
		t.Fatalf("pressure %+v", sample)
	}
}

func TestResourceMonitorPeakPressureKeepsATransient(t *testing.T) {
	monitor := &ResourceMonitor{sampler: NewResourceSampler(func() []int { return nil })}
	pressures := []string{"normal", "critical", "normal"}
	i := 0
	monitor.sampler.sysmem = func() (macx.SystemMemory, error) {
		p := pressures[min(i, len(pressures)-1)]
		i++
		return macx.SystemMemory{Pressure: p}, nil
	}
	monitor.sampler.Sample()
	if peak := monitor.PeakPressure(); peak != "normal" {
		t.Fatalf("peak %q, want normal", peak)
	}
	monitor.sampler.Sample()
	monitor.sampler.Sample()
	sample := monitor.Snapshot("first_request")
	if sample.Pressure != "normal" {
		t.Fatalf("instantaneous %q, want normal again", sample.Pressure)
	}
	if peak := monitor.PeakPressure(); peak != "critical" {
		t.Fatalf("peak %q, want the transient critical kept", peak)
	}
}
