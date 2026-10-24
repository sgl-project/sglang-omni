# Notices for JapaneseInput

JapaneseInput links these third-party components at the exact revisions below.
Each component's license text is included verbatim in this directory.

| Component | Pinned revision | License | File |
| --- | --- | --- | --- |
| azooKey/AzooKeyKanaKanjiConverter | v0.11.2 (`80b8204f1cdfb364bb2ed355cf52c7ebb2519a0c`) | MIT | `LICENSE-AzooKeyKanaKanjiConverter-MIT.txt` |
| azooKey dictionary data | `azooKey_dictionary_storage` `832fbb0d3039dfaa4b2183956f3d96f6b07eec4d` (v2.4.2-5), `Dictionary/` bundled by this package | Apache-2.0 | `LICENSE-azooKey-dictionary-Apache-2.0.txt` |
| ensan-hcl/swift-tokenizers | 0.0.1 (`4a606f66e0cc4d7d9f0197649e812f7fc86a4c34`) | Apache-2.0 | `LICENSE-swift-tokenizers-Apache-2.0.txt` |
| johnmai-dev/Jinja | 1.1.0 (`9c0bbbc7f553932661fc791d2c5664311af09583`) | MIT | `LICENSE-Jinja-MIT.txt` |
| apple/swift-algorithms | 1.2.1 | Apache-2.0 | `LICENSE-swift-algorithms-Apache-2.0.txt` |
| apple/swift-argument-parser | 1.4.0 | Apache-2.0 | `LICENSE-swift-argument-parser-Apache-2.0.txt` |
| apple/swift-collections | 1.7.3 | Apache-2.0 | `LICENSE-swift-collections-Apache-2.0.txt` |
| apple/swift-numerics | 1.1.1 | Apache-2.0 | `LICENSE-swift-numerics-Apache-2.0.txt` |

None of the pinned repositories above contains a `NOTICE` file.

## Tokenizer data arriving via AzooKeyKanaKanjiConverter

`KanaKanjiConverterModule` unconditionally bundles a 528 KB tokenizer under
`Sources/EfficientNGram/tokenizer/`. That directory's README states:
"This tokenizer data is from [ku-nlp/gpt2-small-japanese-char](https://huggingface.co/ku-nlp/gpt2-small-japanese-char),
following CC BY-SA 4.0 License." The license text is at
https://creativecommons.org/licenses/by-sa/4.0/. The license statement above is
sourced from the pinned revision's in-tree README.

Sources: https://github.com/azooKey/AzooKeyKanaKanjiConverter,
https://github.com/ensan-hcl/azooKey_dictionary_storage,
https://github.com/ensan-hcl/swift-tokenizers, https://github.com/johnmai-dev/Jinja,
https://github.com/apple/swift-algorithms, https://github.com/apple/swift-argument-parser,
https://github.com/apple/swift-collections, https://github.com/apple/swift-numerics.
