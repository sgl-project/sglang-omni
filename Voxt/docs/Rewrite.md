# Rewrite

This document explains Voxt's rewrite feature: what it does, how it differs from standard transcription and translation, and how to configure it well.

## Overview

Rewrite is the voice-driven prompt / rewrite flow in Voxt.

Default shortcut: `fn+control`

It turns your speech into an instruction, then uses the configured rewrite model to:

- rewrite selected text
- generate new text from a spoken instruction
- keep a stable answer card when direct text insertion is not appropriate

## Two Main Modes

### Rewrite Selected Text

If text is currently selected, your spoken instruction is applied to that source text.

Examples:

- “Make this shorter and smoother.”
- “Rewrite this in a more formal tone.”
- “Turn this into bullet points.”

### Generate From Voice Prompt

If nothing is selected, your spoken instruction becomes the prompt itself.

Examples:

- “Help me write a 200-word self introduction.”
- “Draft a short project update for my manager.”

## Input Scope And Privacy

Context Enhancement (window text, UI structure and screenshots) has been removed. Rewrite no longer scans the foreground window or attaches screenshots. Requests use the spoken instruction, explicitly selected source text and existing conversation as text inputs. Select a message or provide its contents before asking to reply to it; the model cannot see your screen.

Screen Recording and Input Monitoring are no longer requested. Accessibility remains necessary for global shortcuts and automatic insertion. App Branch routing, dictionary features and selected-text operations remain; this does not mean all cross-app text access has been removed. Custom prompts are preserved, so review any old rules that depend on screenshots.

## Model And Provider Selection

Rewrite follows the rewrite model/provider settings in the main window's `Model` page.

Current rewrite providers:

- local Custom LLM
- Remote LLM

Rewrite does not use ASR provider settings for the generation step; ASR is only used to convert your speech into the rewrite instruction.

## Prompt Behavior

Rewrite uses the dedicated rewrite prompt.

- you can customize it in the main window
- the spoken instruction becomes part of the rewrite request
- selected text, if present, is passed in as source material

Voxt also applies the normal enhancement chain to the dictated instruction before the rewrite step when appropriate.

## App Enhancement And Rewrite

When `App Enhancement` is enabled, rewrite can become context-aware.

<video src="https://github.com/user-attachments/assets/72e361eb-45c3-4f54-ac66-78d4787c7253" controls preload="none" width="100%"></video>

- Voxt can switch prompts and rules based on the current app or browser URL
- the same rewrite shortcut can behave differently in coding tools, chat apps, email, or docs
- this is especially useful when you want different rewrite tone, formatting, or terminology by context

`App Enhancement` does not replace rewrite itself. It changes the prompt routing and context that rewrite uses.

## Rewrite Answer Card

Rewrite can keep the result visible in an answer card instead of relying only on direct text insertion.

This is useful when:

- the current target is not writable
- the generated result is long
- you want to review the answer before using it

## Tips

- Be explicit about style, length, and output format
- For selected-text rewrite, mention what should change, not just the topic
- For prompt-style generation, say what you want written and any constraints

## Notes

- Rewrite is a separate flow from normal transcription and translation
- It can reuse the same local or remote LLM infrastructure, but with its own prompt and output behavior
- Result quality depends strongly on the chosen model and the clarity of the spoken instruction
