package updates

import "testing"

func TestBuildBackend(t *testing.T) {
	for _, test := range []struct {
		name           string
		hasFeed        bool
		checksReleases bool
		want           Backend
	}{
		{"feed", true, false, FeedUpdates},
		{"feed wins over the fallback", true, true, FeedUpdates},
		{"release without a feed", false, true, GitHubUpdates},
		{"development build", false, false, NoUpdates},
	} {
		t.Run(test.name, func(t *testing.T) {
			if got := BuildBackend(test.hasFeed, test.checksReleases); got != test.want {
				t.Errorf("BuildBackend(%v, %v) = %v, want %v", test.hasFeed, test.checksReleases, got, test.want)
			}
		})
	}
}
