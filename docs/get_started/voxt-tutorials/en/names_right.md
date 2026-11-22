# Get names and terms spelled right

[中文](../names_right.md)

Voxt keeps writing a colleague's name or a product term wrong, and you fix it by hand every time. Add it to the dictionary once and it sticks.

1. Open Voxt's main window, go to **Dictionary** in the sidebar, and click **Add term**.
2. **Term** is the right spelling, such as "Kubernetes"; **Aliases** is what speech recognition hears instead, such as "koo ber netties". Save.
3. Say the same sentence again and the name comes out right. Terms go to the speech model as words to expect, and the language model spells them as written here.

With many terms, click **Import…** on the **Dictionary** page to import a CSV or a text list at once. When a spelling applies only in certain situations, write the rule in the **personal dictionary**'s Markdown note, which goes to the language model with each request. The dictionary applies to [Transcribe](transcribe.md), [Translate](translate.md) and [Rewrite](rewrite.md).

## For agents

> Following this page, open Voxt's Dictionary page and wait for me to give the right spelling and the common mishearing, then add the term. Confirm with one test sentence containing the name and tell me both results, before and after.
