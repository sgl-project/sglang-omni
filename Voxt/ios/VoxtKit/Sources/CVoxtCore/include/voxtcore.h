// The C ABI of voxtcore (ios/voxtcore/cabi), a static library built by
// scripts/build-core.sh. Every call is JSON in and JSON out:
// {"result": ...} or {"error": {"code": ..., "message": ...}}.
#ifndef VOXTCORE_H
#define VOXTCORE_H

#ifdef __cplusplus
extern "C" {
#endif

// Runs method with its JSON params and returns the reply, which the caller
// frees with VoxtCoreFree. It blocks until the call ends: call it off the
// main thread. id names the call for VoxtCoreCancel ("" for none).
char *VoxtCoreCall(const char *id, const char *method, const char *params);

// Cancels the running call id; returns 1 when there was one.
int VoxtCoreCancel(const char *id);

// Frees a reply of VoxtCoreCall.
void VoxtCoreFree(char *reply);

#ifdef __cplusplus
}
#endif

#endif
