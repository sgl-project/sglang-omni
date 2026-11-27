# Translate with Voxt

[中文](../translate.md)

Voxt can translate selected text, or a sentence you say, into the language you choose. First [set up a language model](byok.md). Translating speech also needs a [downloaded speech model](start.md#download-a-speech-model).

1. Open **Translate** in the Voxt sidebar and choose the **Target language** and **Language model**. Translate defaults to English, or to Chinese when your main language is English.
2. **Translate what you say.** Place the cursor in an empty text field with nothing selected, tap `⇧ Shift` + Right `⌥ Option`, say “please confirm the time by Friday”, and press the same shortcut again. The translation appears at the cursor.

   <div class="vx-anim" data-scene="translate-voice" data-lang="en"></div>

3. **Translate selected text.** Select some text, press `⇧ Shift` + Right `⌥ Option`, and wait: the translation replaces the selection.

   <div class="vx-anim" data-scene="translate-select" data-lang="en"></div>

If step 2 starts translating text that is already there, deselect it and try again. The **Translate** page also has a **fallback model**, used when the first one fails, and **Your rules**, such as “don't translate names”. The current shortcut is the one on the **Translate** page.

## For agents

> Following this page, set Voxt's Translate target language and choose the language model I set up. In a new empty document, type “please confirm the time by Friday”, select it and test translating the selection, keeping the original for me to compare. For translating speech, stop where I need to start speaking.
