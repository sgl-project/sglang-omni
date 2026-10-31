#ifndef VOXT_CPINYIN_H
#define VOXT_CPINYIN_H
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif
typedef struct VoxtPinyin VoxtPinyin;
VoxtPinyin *voxt_pinyin_create(const char *path);
void voxt_pinyin_destroy(VoxtPinyin *engine);
// Syllables or initials (简拼) use '-' separators. Returns the candidate count,
// -1 for invalid syllables, -2 for engine error.
int voxt_pinyin_convert(VoxtPinyin *engine, const char *syllables);
// Owned by the engine; valid until the next conversion or destruction.
const char *voxt_pinyin_candidate(const VoxtPinyin *engine, size_t index);
// Sum of the log10 scores of the best sentence from the last conversion.
double voxt_pinyin_score(const VoxtPinyin *engine);
// How many syllables, from the first, the candidate replaces.
size_t voxt_pinyin_candidate_length(const VoxtPinyin *engine, size_t index);
// The dictionary reading the candidate came from ("ni-hao" for 你好 typed as
// n-h).
const char *voxt_pinyin_candidate_reading(const VoxtPinyin *engine,
                                          size_t index);
// Space-separated syllables that initials (简拼) expand to. None until set.
void voxt_pinyin_set_syllables(VoxtPinyin *engine, const char *syllables);
// Nonzero: z/zh, c/ch, s/sh readings match each other. Off when created.
void voxt_pinyin_set_fuzzy(VoxtPinyin *engine, int fuzzy);
#ifdef __cplusplus
}
#endif
#endif
