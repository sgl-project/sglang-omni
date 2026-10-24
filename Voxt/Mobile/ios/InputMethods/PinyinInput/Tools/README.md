# Regenerating the pinyin data

```sh
cd ios/PinyinInput/Tools
mkdir -p source
rev=2d62adc58635968c94522cfdc4103cef139e58dc
for f in dicts/zi.dict.yaml dicts/jichu.dict.yaml LICENSE; do
  curl -L -o "source/$(basename $f)" "https://raw.githubusercontent.com/amzxyz/rime-wanxiang/$rev/$f"
done
echo $rev > source/COMMIT
python3 -m unittest test_make_data
python3 -I make_data.py            # writes out/pinyin-data.txt and out/manifest.json
```

The output must match `Sources/PinyinInput/Resources/Notices/DictionaryManifest.json`
(`bytes`, `output_sha256`). Copy `out/pinyin-data.txt` to `Sources/PinyinInput/Resources/`
and update `PinyinConverter.dataSHA256` and `dataBytes` if you change the pruning.
