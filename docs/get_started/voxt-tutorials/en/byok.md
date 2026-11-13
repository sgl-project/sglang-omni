# Set up Voxt with your own API key (BYOK)

[中文](../byok.md)

Save and test a language model service in Voxt, and Translate, Rewrite and the optional clean-up of transcriptions can use it. The local speech model turns your voice into text; the language model then works on that text for the feature you use.

## Get the values from your provider

Open the console of a provider you can use, go to **API keys**, and create or copy a key for Voxt. If your team provides the service, get the key through the channel your team uses for secrets. For another OpenAI-compatible service, also get the **model ID** and the **API address** from the same provider. All three must belong to the same setup.

## Put the key into Voxt

1. Open **Language models** in the Voxt sidebar (or click **Open Language models** in the sidebar's “No language model” notice, or **Set up a language model** on the **Model** step of the setup guide). Choose your provider; for a service compatible with OpenAI Chat Completions, choose the **OpenAI** card.

   <div class="vx-anim" data-scene="llm-open" data-lang="en"></div>

2. Click the **API Key** field and press `⌘V` to paste the key. It shows as dots, and when you save, it goes into this Mac's Keychain, never into Voxt's settings or logs.
3. For the provider's own service, leave **Address** as it is. For another OpenAI-compatible service, enter its API root, usually ending in `/v1`. It must start with `https://` (`http://` only for a server on this Mac).
4. Choose one of the provider's models under **Model**. If yours is not listed, choose **Other model…** and paste the exact model ID into the field that appears; it is case-sensitive.

   <div class="vx-anim" data-scene="llm-fill" data-lang="en"></div>

   In some versions these are labelled **Custom…** instead of **Other model…**, and **Endpoint** instead of **Address**; **Endpoint** takes the full Chat Completions URL, ending in `/chat/completions`. Fill in what your version shows. You can leave the generation settings as they are for now.

5. Click **Test connection** (**Test** in some versions). When it says **Connection OK**, click **Save**. The test calls the service you set up. If a custom service uses a different protocol, check the API format under **Advanced** against your provider's documentation.

   <div class="vx-anim" data-scene="llm-test" data-lang="en"></div>

## Use it in a feature

Open **Translate** or **Rewrite** in the sidebar and choose the saved setup under **Language model**. In **Transcribe**, choose a language model only if you want fillers removed or the text cleaned up. Try it once with a harmless sentence, such as “please confirm the time by Friday”.

<div class="vx-anim" data-scene="llm-use" data-lang="en"></div>

## If the test fails

| Message | What to do |
|---|---|
| The remote service refused the API key. | The key is wrong, expired or revoked. Copy the whole key again and paste it |
| Paste the service's API key above, then test again. | No key yet |
| The remote model's address is not allowed. Use https://, or http:// for an address on this Mac. | Make the address start with `https://` |
| Model not found, 404 | Check that the model ID matches exactly and the address ends in `/v1` |
| The remote service refused the request, or rate limits | Check the key's access and credit in the provider's console |

Fix it and click **Test connection** again.

## For agents

> Following this page, open Voxt's Language models settings and choose the preset or a custom setup for the provider and API format I give you. Fill in the non-secret fields, then stop at the API Key field and let me paste the key myself from a secure place. Do not read, repeat or screenshot that field. When I am done, test the connection with Voxt's test button, save, and choose this setup for the features I name. Confirm it works with a harmless example and report only the status.
