//go:build !darwin

package profile

import "context"

func ReadHostSample(context.Context) HostSample {
	return HostSample{
		Thermal:     Unavailable[string]("pmset thermal state is available only on Darwin"),
		PowerSource: Unavailable[string]("pmset power source is available only on Darwin"),
	}
}
