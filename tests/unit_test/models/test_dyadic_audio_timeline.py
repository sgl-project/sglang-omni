# SPDX-License-Identifier: Apache-2.0
"""Both parties' audio on a Session's input timeline: windows at 16 kHz, reply placement and interrupts."""

import numpy as np
from scipy.signal import resample_poly

from sglang_omni.models.dyadic_audio.timeline import DyadicAudio, ReplyTrack, TimedTrack

UNIT_MS = 80


def test_a_16_khz_track_reads_back_its_samples_with_silence_where_none_arrived():
    track = TimedTrack()
    track.append(100, np.full(800, 0.25, dtype=np.float32), eos=False)
    track.append(200, np.full(800, 0.5, dtype=np.float32), eos=False)
    window = track.window(1000, 4000)
    assert track.end == 4000
    assert not window[:600].any()
    assert window[600:1400].tolist() == [0.25] * 800
    assert not window[1400:2200].any()
    assert window[2200:3000].tolist() == [0.5] * 800
    assert not window[3000:].any()


def test_a_later_chunk_overwrites_an_earlier_one_and_dropped_audio_reads_as_silence():
    track = TimedTrack()
    track.append(0, np.full(3200, 0.25, dtype=np.float32), eos=False)
    track.append(100, np.full(800, 0.5, dtype=np.float32), eos=True)
    assert track.window(1600, 800).tolist() == [0.5] * 800
    assert track.is_ended
    track.drop_before(2400)
    assert track.offset == 2390
    assert not track.window(0, 2390).any()
    assert track.window(2400, 800).tolist() == [0.25] * 800


def test_24_khz_units_read_back_as_one_resampling_of_the_whole_stream():
    rng = np.random.default_rng(0)
    whole = rng.standard_normal(24 * 2000).astype(np.float32)
    track = TimedTrack(24000)
    unit = 24 * UNIT_MS
    windows = []
    for index in range(0, whole.size // unit):
        track.append(
            index * UNIT_MS, whole[index * unit : (index + 1) * unit], eos=False
        )
        if (index + 1) % 10 == 0:
            start = (index - 9) * UNIT_MS * 16
            # A window reads once the stream covers it and a little past, as a block waits for the next unit.
            windows.append((start, track.window(start, 10 * UNIT_MS * 16)))
            track.drop_before(start + 10 * UNIT_MS * 16)
        else:
            pass
    reference = resample_poly(whole, 2, 3)
    assert track.end == 32000
    for start, window in windows[:-1]:
        assert (
            np.abs(window[40:-40] - reference[start + 40 : start + 12800 - 40]).max()
            < 1e-4
        )


def test_replies_follow_on_and_an_interrupt_cuts_them_and_drops_their_later_audio():
    track = ReplyTrack.at_rate(24000)
    track.append_following(3, np.ones(2400, dtype=np.float32), cursor=0, boundary=12800)
    track.append_following(3, np.ones(2400, dtype=np.float32), cursor=0, boundary=12800)
    track.append_following(4, np.ones(2400, dtype=np.float32), cursor=0, boundary=12800)
    assert [
        (reply.turn, reply.start, track.length(reply)) for reply in track.replies
    ] == [
        (3, 0, 3200),
        (4, 12800, 1600),
    ]
    assert track.turn_at(0, 800) == 3 and track.turn_at(4000, 800) is None
    track.interrupt(3, at=1600)
    track.append_following(3, np.ones(2400, dtype=np.float32), cursor=0, boundary=12800)
    assert [(reply.turn, track.length(reply)) for reply in track.replies] == [
        (3, 1600),
        (4, 1600),
    ]


def test_pending_replies_wait_for_their_length_or_their_end():
    track = ReplyTrack.at_rate(16000)
    track.append_pending(1, np.ones(500, dtype=np.float32))
    track.append_pending(2, np.ones(5000, dtype=np.float32))
    track.place(cursor=640, boundary=640, min_length=1000)
    assert [reply.start for reply in track.replies] == [None, None]
    track.finish(1)
    track.place(cursor=640, boundary=640, min_length=1000)
    assert [reply.start for reply in track.replies] == [640, 1280]
    track.interrupt(None, at=1280)
    assert [track.length(reply) for reply in track.replies] == [500, 0]


def test_streaming_units_hold_a_window_back_until_they_reach_it():
    audio = DyadicAudio.at_rates(user_rate_hz=24000, speech_rate_hz=24000)
    assert not audio.awaits_units(12800)
    audio.units.append(0, np.full(24 * 160, 0.5, dtype=np.float32), eos=False)
    assert audio.awaits_units(12800) and not audio.awaits_units(2560)
    audio.units.append(160, np.zeros(0, dtype=np.float32), eos=True)
    assert not audio.awaits_units(12800)
    assert audio.speech_end() == 2560
    assert np.allclose(audio.speech_window(0, 2560)[100:-100], 0.5, atol=1e-3)


def test_a_declared_reply_stream_holds_a_window_back_before_the_first_unit_too():
    audio = DyadicAudio.at_rates(
        user_rate_hz=24000, speech_rate_hz=24000, reply_stream=True
    )
    assert audio.awaits_units(1) and audio.awaits_units(12800)
    audio.units.append(0, np.zeros(24 * 160, dtype=np.float32), eos=False)
    assert audio.awaits_units(12800) and not audio.awaits_units(2560)
    audio.units.append(160, np.zeros(0, dtype=np.float32), eos=True)
    assert not audio.awaits_units(12800)
    # A stream that ends before its first unit holds nothing back.
    ended = DyadicAudio.at_rates(speech_rate_hz=24000, reply_stream=True)
    ended.units.is_ended = True
    assert not ended.awaits_units(12800)
