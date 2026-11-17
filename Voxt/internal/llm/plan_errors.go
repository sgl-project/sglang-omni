package llm

import (
	"context"
	"strings"
)

// Credential is what a request to a signed-in service sends.
type Credential struct{ Token, AccountID string }

// TokenSource gives a signed-in service's credential. stale, when not "",
// is the token the service just refused.
type TokenSource func(ctx context.Context, stale string) (Credential, error)

// WithTokens makes o ask src for its credential on every request, for a
// preset with SignIn.
func (o *OpenAI) WithTokens(src TokenSource) *OpenAI {
	o.tokens = src
	return o
}

// SignInError is the error of a service whose sign-in ended.
func SignInError(provider string) *Error { return newErr(CodeSignIn, "provider", provider) }

// planError is the failure of a signed-in plan that the status alone would
// call something else (HTTPError); nil when it is not one. The codes are
// the ChatGPT backend's, as the Codex CLI reads them.
func planError(provider string, status int, detail string) *Error {
	low := strings.ToLower(detail)
	switch {
	case status == 401:
		return newErr(CodeSignIn, "provider", provider)
	case strings.Contains(low, "usage_limit_reached"), strings.Contains(low, "insufficient_quota"):
		return newErr(CodeUsageLimit, "provider", provider, "detail", clip(detail, 300))
	case strings.Contains(low, "usage_not_included"), status == 403:
		return newErr(CodeNotEligible, "provider", provider, "detail", clip(detail, 300))
	}
	return nil
}
