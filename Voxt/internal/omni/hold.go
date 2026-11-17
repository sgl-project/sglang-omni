package omni

import "context"

// HoldTarget names one model server to keep loaded (meeting: ASR and Silero VAD).
type HoldTarget struct {
	Spec      Spec
	Dir       string
	Draft     string // the speculative draft's folder, "" none
	DraftBits int    // 4, 6 or 8; 0 means 8
}

// Hold keeps the given models' servers loaded until release. While held,
// idle timeout does not unload them (a background lease per server).
func (rt *Runtime) Hold(ctx context.Context, targets ...HoldTarget) (func(), error) {
	var leases []*lease
	for _, t := range targets {
		l, err := rt.acquireDraft(ctx, t.Spec, t.Dir, t.Draft, t.DraftBits)
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
