//go:build !darwin

package profile

// Signposts is unavailable off macOS; profiling and JSON output still work.
type Signposts struct{}

func NewSignposts() *Signposts { return &Signposts{} }

func (s *Signposts) UnavailableReason() string {
	return "os_signpost is available only on Darwin"
}

func (s *Signposts) Begin(string) func() { return func() {} }
