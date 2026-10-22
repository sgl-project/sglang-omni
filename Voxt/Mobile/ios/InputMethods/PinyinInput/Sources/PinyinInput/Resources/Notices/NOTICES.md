# Pinyin input sources

Engine: McBopomofo 3.1.1, commit f5ba010ce8795d283ee336ca7d16380f200bd2ec
(https://github.com/openvanilla/McBopomofo). ParselessLM, ParselessPhraseDB,
MemoryMappedFile, reading_grid and language_model are copied without modification
into ios/McBopomofoEngine, shared with the Zhuyin keyboard; each keeps its MIT
notice, and the root MIT text is McBopomofo-MIT.txt.

Dictionary: rime-wanxiang (https://github.com/amzxyz/rime-wanxiang), commit
2d62adc58635968c94522cfdc4103cef139e58dc, dicts/zi.dict.yaml and
dicts/jichu.dict.yaml, licensed CC-BY-4.0 (rime-wanxiang-CC-BY-4.0.txt).
Modified: tones removed (ü written as v), rows merged by toneless reading,
pruned to all characters plus 250,000 words (162,000 of them two-character),
weights converted to log10 probabilities, and written in McBopomofo's sorted
format. Tools/make_data.py does this; DictionaryManifest.json records the input,
script and output SHA256 values.
