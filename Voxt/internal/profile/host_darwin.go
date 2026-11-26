//go:build darwin

package profile

import (
	"context"
	"os/exec"
	"strings"
)

func pmset(ctx context.Context, argument string) (string, error) {
	output, err := exec.CommandContext(ctx, "/usr/bin/pmset", "-g", argument).CombinedOutput()
	return strings.TrimSpace(string(output)), err
}

func powerSource(output string) string {
	for _, line := range strings.Split(output, "\n") {
		if !strings.Contains(line, "Now drawing from") {
			continue
		}
		start := strings.Index(line, "'")
		end := strings.LastIndex(line, "'")
		if start >= 0 && end > start {
			return line[start+1 : end]
		}
	}
	return ""
}

func ReadHostSample(ctx context.Context) HostSample {
	var sample HostSample
	thermal, err := pmset(ctx, "therm")
	if err != nil {
		sample.Thermal = Unavailable[string]("pmset -g therm: " + err.Error())
	} else if thermal == "" {
		sample.Thermal = Unavailable[string]("pmset -g therm returned no state")
	} else {
		sample.Thermal = Available(thermal)
	}
	battery, err := pmset(ctx, "batt")
	source := powerSource(battery)
	if err != nil || source == "" {
		sample.PowerSource = Unavailable[string]("pmset -g batt did not report a power source")
	} else {
		sample.PowerSource = Available(source)
	}
	return sample
}
