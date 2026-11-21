//go:build darwin

package profile

import "testing"

func TestSignpostsAvailable(t *testing.T) {
	signposts := NewSignposts()
	if reason := signposts.UnavailableReason(); reason != "" {
		t.Fatal(reason)
	}
	end := signposts.Begin("profile_test")
	end()
	end()
}
