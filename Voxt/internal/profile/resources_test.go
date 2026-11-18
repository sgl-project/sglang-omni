package profile

import (
	"errors"
	"testing"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/macx"
)

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
}
