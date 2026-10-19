# SPDX-License-Identifier: Apache-2.0
"""Single source of truth for the Cookbook sidebar order (PR sgl-project/sglang-omni#2876).

Everything the lint needs is data, not logic. When the rules change, edit this
file; scripts/check_cookbook_order.py does not need to change.

  * If the #1081 roadmap changes (sgl-project/sglang-omni#1081): update
    ROADMAP_ORDER to the new **Focus -> Supporting** order.
  * If Hugging Face download counts change: run
    `python3 scripts/check_cookbook_order.py --refresh-downloads`, which
    re-queries the HF API, rewrites DOWNLOADS_DESC, and prints the new expected
    order. Then update docs/index.rst if the rank changed.
  * New model: add the page to ROADMAP_ORDER (roadmap model, at its #1081
    position) or to HF_REPO + DOWNLOADS_DESC (non-roadmap model, with its HF
    download count), then place the docs/index.rst entry where the count
    implies. `--print-expected` prints the full expected order.
"""

# Models listed under **Focus -> Supporting** in sgl-project/sglang-omni#1081,
# in that exact order. Qwen3-TTS Base and CustomVoice share one cookbook page.
ROADMAP_ORDER = [
    "cookbook/qwen3_tts.md",
    "cookbook/moss_tts.md",
    "cookbook/whisper_asr.md",
    "cookbook/nemotron3_5_asr.md",
    "cookbook/moss_transcribe_diarize.md",
    "cookbook/fishaudio_s2_pro.md",
    "cookbook/cosmos3_nano.md",
    "cookbook/qwen3_omni.md",
    "cookbook/fun_cosyvoice3.md",
    "cookbook/personaplex.md",
    "cookbook/minicpm_o.md",
    "cookbook/llada2_uni.md",
]

# Hugging Face download counts for every non-roadmap cookbook entry, in
# descending order. Refresh with:
#   python3 scripts/check_cookbook_order.py --refresh-downloads
DOWNLOADS_DESC = {
    "cookbook/qwen3_asr.md": 1_151_241,
    "cookbook/fun_asr.md": 158_926,
    "cookbook/higgs_tts.md": 120_276,
    "cookbook/moss_tts_local.md": 117_715,
    "cookbook/nemotron_diarization.md": 98_684,
    "cookbook/fishaudio_drama3.md": 9_620,
    "cookbook/minimax_music3.md": 6_207,
    "cookbook/ming_omni.md": 5_430,
    "cookbook/nemotron_voicechat.md": 4_864,
    "cookbook/auk.md": 2_486,
    "cookbook/arkasr.md": 2_411,
    "cookbook/zonos2.md": 2_359,
    "cookbook/voxtral_tts.md": 1_589,
    "cookbook/dots_tts.md": 755,
    "cookbook/ming_tts.md": 319,
}

# Cookbook page -> Hugging Face repo id, used by --refresh-downloads. For a
# new non-roadmap model, add the page here so the refresh can find its count.
HF_REPO = {
    "cookbook/qwen3_asr.md": "Qwen/Qwen3-ASR-1.7B",
    "cookbook/fun_asr.md": "FunAudioLLM/Fun-ASR-Nano-2512-hf",
    "cookbook/higgs_tts.md": "bosonai/higgs-audio-v3-tts-4b",
    "cookbook/moss_tts_local.md": "OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5",
    "cookbook/nemotron_diarization.md": "nvidia/Nemotron-3-Diarization",
    "cookbook/minimax_music3.md": "MiniMaxAI/MiniMax-Music3",
    "cookbook/fishaudio_drama3.md": "fishaudio/drama-3",
    "cookbook/ming_omni.md": "inclusionAI/Ming-flash-omni-2.0",
    "cookbook/nemotron_voicechat.md": "nvidia/NVIDIA-NemotronLabs-VoiceChat-11B",
    "cookbook/auk.md": "tencent/AuK",
    "cookbook/arkasr.md": "AutoArk-AI/ARK-ASR-3B",
    "cookbook/zonos2.md": "Zyphra/zonos2",
    "cookbook/voxtral_tts.md": "mistralai/Voxtral-4B-TTS-2603",
    "cookbook/dots_tts.md": "dots-studio/dots.tts-mf",
    "cookbook/ming_tts.md": "inclusionAI/Ming-omni-tts-16.8B-A3B",
}

# When DOWNLOADS_DESC was last refreshed (kept in sync by --refresh-downloads).
DOWNLOADS_QUERIED_AT = "2026-10"
