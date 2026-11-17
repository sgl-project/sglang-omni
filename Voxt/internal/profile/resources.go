package profile

import (
	"context"
	"os"
	"sync"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/macx"
)

const (
	gpuMemoryReason = "Apple silicon has unified memory; proc_pid_rusage reports physical footprint including Metal buffers, not a separate GPU allocation"
	gpuTimeReason   = "no supported per-process GPU time counter is available to this harness"
	energyReason    = "energy is not collected: powermetrics would require elevated privileges"
)

func ReadMemorySample() MemorySample {
	memory, err := macx.ReadSystemMemory()
	if err != nil {
		reason := err.Error()
		return MemorySample{
			Pressure: Unavailable[string](reason), AvailablePercent: Unavailable[int64](reason),
			SwapUsedBytes: Unavailable[int64](reason), SwapTotalBytes: Unavailable[int64](reason),
			Swapins: Unavailable[int64](reason), Swapouts: Unavailable[int64](reason),
		}
	}
	sample := MemorySample{Pressure: Available(memory.Pressure)}
	if memory.AvailablePercentValid {
		sample.AvailablePercent = Available(int64(memory.AvailablePercent))
	} else {
		sample.AvailablePercent = Unavailable[int64]("kern.memorystatus_level is unavailable")
	}
	if memory.SwapValid {
		sample.SwapUsedBytes = Available(memory.SwapUsedBytes)
		sample.SwapTotalBytes = Available(memory.SwapTotalBytes)
	} else {
		sample.SwapUsedBytes = Unavailable[int64]("vm.swapusage is unavailable")
		sample.SwapTotalBytes = Unavailable[int64]("vm.swapusage is unavailable")
	}
	if memory.SwapinsValid {
		sample.Swapins = Available(memory.Swapins)
	} else {
		sample.Swapins = Unavailable[int64]("swapins counter is unavailable")
	}
	if memory.SwapoutsValid {
		sample.Swapouts = Available(memory.Swapouts)
	} else {
		sample.Swapouts = Unavailable[int64]("swapouts counter is unavailable")
	}
	return sample
}

// ResourceSampler takes instantaneous process samples. Its peak is the
// largest observed sum over the harness and native servers, not a kernel
// lifetime high-water mark.
type ResourceSampler struct {
	mu            sync.Mutex
	processes     func() []int
	memory        func(int) (macx.ProcMemory, error)
	cpu           func(int) (macx.ProcCPU, error)
	peakRSS       int64
	peakFootprint int64
	memorySamples int
	memoryError   string
	latestCPU     map[int]macx.ProcCPU
	cpuError      string
}

func NewResourceSampler(processes func() []int) *ResourceSampler {
	return &ResourceSampler{
		processes: processes, memory: macx.Memory, cpu: macx.CPUTime,
		latestCPU: map[int]macx.ProcCPU{},
	}
}

func uniquePIDs(processes []int) []int {
	seen := map[int]bool{}
	var unique []int
	for _, process := range processes {
		if process > 0 && !seen[process] {
			seen[process] = true
			unique = append(unique, process)
		}
	}
	return unique
}

func (s *ResourceSampler) Sample() {
	s.mu.Lock()
	defer s.mu.Unlock()
	var rss, footprint int64
	processes := uniquePIDs(s.processes())
	memoryOK := len(processes) > 0
	for _, process := range processes {
		memory, err := s.memory(process)
		if err != nil {
			memoryOK = false
			if s.memoryError == "" {
				s.memoryError = err.Error()
			}
		} else {
			rss += memory.Resident
			footprint += memory.Footprint
		}
		cpu, err := s.cpu(process)
		if err != nil {
			if s.cpuError == "" {
				s.cpuError = err.Error()
			}
		} else {
			s.latestCPU[process] = cpu
		}
	}
	if memoryOK {
		s.memorySamples++
		s.peakRSS = max(s.peakRSS, rss)
		s.peakFootprint = max(s.peakFootprint, footprint)
	}
}

func (s *ResourceSampler) report(start, end MemorySample, hostStart, hostEnd HostSample) ResourceReport {
	s.mu.Lock()
	defer s.mu.Unlock()
	report := ResourceReport{
		GPUMemoryBytes: Unavailable[int64](gpuMemoryReason),
		GPUTimeMS:      Unavailable[float64](gpuTimeReason),
		EnergyJoules:   Unavailable[float64](energyReason),
		MemoryStart:    start,
		MemoryEnd:      end,
		HostStart:      hostStart,
		HostEnd:        hostEnd,
	}
	if s.memorySamples == 0 {
		reason := s.memoryError
		if reason == "" {
			reason = "no process memory sample succeeded"
		}
		report.PeakRSSBytes = Unavailable[int64](reason)
		report.PeakPhysicalFootprint = Unavailable[int64](reason)
	} else {
		report.PeakRSSBytes = Available(s.peakRSS)
		report.PeakPhysicalFootprint = Available(s.peakFootprint)
	}
	if len(s.latestCPU) == 0 {
		reason := s.cpuError
		if reason == "" {
			reason = "no process CPU sample succeeded"
		}
		report.CPUUserMS = Unavailable[float64](reason)
		report.CPUSystemMS = Unavailable[float64](reason)
	} else {
		var user, system time.Duration
		for _, cpu := range s.latestCPU {
			user += cpu.User
			system += cpu.System
		}
		report.CPUUserMS = Available(milliseconds(user))
		report.CPUSystemMS = Available(milliseconds(system))
	}
	report.SwapinsDelta = counterDelta(start.Swapins, end.Swapins)
	report.SwapoutsDelta = counterDelta(start.Swapouts, end.Swapouts)
	return report
}

func counterDelta(start, end OptionalInt64) OptionalInt64 {
	if start.Value == nil || end.Value == nil {
		return Unavailable[int64]("swap counters are unavailable at the start or end")
	}
	delta := *end.Value - *start.Value
	return Available(delta)
}

// ResourceMonitor samples until Stop. Processes that exit before Stop keep
// their last successful CPU sample and contribute to earlier peaks.
type ResourceMonitor struct {
	sampler   *ResourceSampler
	cancel    context.CancelFunc
	done      chan struct{}
	start     MemorySample
	hostStart HostSample
	stopOnce  sync.Once
	report    ResourceReport
}

func StartResourceMonitor(parent context.Context, processes func() []int) *ResourceMonitor {
	ctx, cancel := context.WithCancel(parent)
	monitor := &ResourceMonitor{
		sampler: NewResourceSampler(processes), cancel: cancel, done: make(chan struct{}),
		start: ReadMemorySample(), hostStart: ReadHostSample(parent),
	}
	monitor.sampler.Sample()
	go func() {
		defer close(monitor.done)
		ticker := time.NewTicker(100 * time.Millisecond)
		defer ticker.Stop()
		for {
			select {
			case <-ticker.C:
				monitor.sampler.Sample()
			case <-ctx.Done():
				return
			}
		}
	}()
	return monitor
}

func (m *ResourceMonitor) Stop(ctx context.Context) ResourceReport {
	m.stopOnce.Do(func() {
		m.cancel()
		<-m.done
		m.sampler.Sample()
		m.report = m.sampler.report(m.start, ReadMemorySample(), m.hostStart, ReadHostSample(ctx))
	})
	return m.report
}

// SelfAndNative returns the harness pid and the native server pids.
func SelfAndNative(native func() []int) func() []int {
	return func() []int {
		return append([]int{os.Getpid()}, native()...)
	}
}
