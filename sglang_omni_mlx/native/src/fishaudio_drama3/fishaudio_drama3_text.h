// SPDX-License-Identifier: Apache-2.0
// Text rules of the Drama 3 prompt, and splitting long text into generation
// chunks.
#pragma once

#include <string>
#include <vector>

namespace fishaudio_drama3 {

// Throws std::invalid_argument for empty text, text holding a structural
// token, or a speaker tag outside 0..15; location leads the message.
void ValidateText(const std::string &text, const std::string &location);
// text when it starts (after white space) with a speaker tag, else text
// tagged as speaker 0.
std::string WithSpeakerTag(const std::string &text);
// A reference transcript tagged so its tag directly follows "Text: " in the
// prompt.
std::string ReferenceTranscript(const std::string &text);
// Generation chunks in order: about 300 bytes of UTF-8 each, with at most
// three speakers, cut at sentence ends, then clause marks, then spaces; tags,
// paired tags and [bracketed] cues are never cut. Text with nothing to speak
// has no chunks.
std::vector<std::string> SplitText(const std::string &text);

} // namespace fishaudio_drama3
