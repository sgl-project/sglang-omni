package omni

import "context"

// HoldTarget names one model server to keep loaded (meeting: ASR and Silero VAD).
type HoldTarget struct {
	Spec  Spec
	Dir   string
	Draft string // the speculative draft's folder, "" none
}

// Hold keeps the given models' servers loaded until release. While held,
// idle timeout does not unload them (a background lease per server).
func (rt *Runtime) Hold(ctx context.Context, targets ...HoldTarget) (func(), error) {
	var leases []*lease
	for _, t := range targets {
		l, err := rt.acquire(ctx, t.Spec, t.Dir, t.Draft)
		if err != nil {
			for _, x := range leases {
				x.release()
			}
			return nil, err
		}
		leases = append(leases, l)
	}
	return func() {
		for _, l := range leases {
			l.release()
		}
	}, nil
}
