# Voxt

## Register

product

## Users

Mac users dictating into other apps, transcribing meetings, and reviewing transcripts.

## Product Purpose

Capture speech with on-device recognition, offer translation and rewriting through the user's configured services, and retain useful meeting transcripts and summaries.

## Brand Personality

Native, quiet, direct. Follow the existing AudioSmith UI kit and MyGo components.

## Anti-references

Avoid nested cards, extra confirmation steps, decorative accents, and implementation details in ordinary user flows. These constraints follow AGENTS.md, ui_tokens.go, and docs/behavior.md.

## Design Principles

- Respond to recording controls immediately.
- Keep live captions readable without distracting from the meeting.
- Preserve identities across live captions, saved transcripts, and exports.
- Show an anonymous speaker label when a name cannot be confirmed.
- Use the existing theme, type scale, spacing, and components.

## Accessibility & Inclusion

Preserve keyboard access and reduced motion. Support English and Simplified Chinese. Speaker colors supplement visible names and labels; color is never the only identifying cue. Maintain readable contrast in light and dark appearances.
