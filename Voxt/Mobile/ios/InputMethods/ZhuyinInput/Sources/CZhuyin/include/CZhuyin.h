#ifndef VOXT_CZHUYIN_H
#define VOXT_CZHUYIN_H
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif
typedef struct VoxtZhuyin VoxtZhuyin;
VoxtZhuyin *voxt_zhuyin_create(const char *path);
void voxt_zhuyin_destroy(VoxtZhuyin *engine);
// Readings use '-' separators. Returns -1 for invalid readings, -2 for engine
// error.
int voxt_zhuyin_convert(VoxtZhuyin *engine, const char *readings,
                        size_t cursor);
// Result pointer is owned by engine, valid until next conversion or
// destruction.
const char *voxt_zhuyin_candidate(const VoxtZhuyin *engine, size_t index);
#ifdef __cplusplus
}
#endif
#endif
