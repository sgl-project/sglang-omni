// Package updates chooses how a build checks for updates.
package updates

// Backend is an update source.
type Backend uint8

const (
	NoUpdates Backend = iota
	FeedUpdates
	GitHubUpdates
)

// BuildBackend chooses the signed feed when the build has one, else the
// GitHub fallback of a release without one.
func BuildBackend(hasFeed, checksReleases bool) Backend {
	if hasFeed {
		return FeedUpdates
	}
	if checksReleases {
		return GitHubUpdates
	}
	return NoUpdates
}
