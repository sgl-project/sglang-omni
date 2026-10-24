# Settings Shell

Shared settings window shell, navigation, controls, styles, and reusable presentation primitives.

## Responsibilities

- Hosts navigation, search, selection controls, paged lists, and window-level settings layout.
- Provides shared button styles, empty states, icons, preference keys, and UI style constants.
- Keeps shell-level controls reusable across individual settings sections.

`SettingsView` owns navigation, observations and page assembly. Sidebar, header and footer are separate value/callback-driven views; notification and feedback dialogs have dedicated files. Notification selection/expansion state stays in the notification dialog, and feedback URLs are supplied as inputs rather than exposed shell internals.
