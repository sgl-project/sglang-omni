"""Browser AV grouping preserves the decoder's delayed video timeline."""

import base64
import importlib.util
import json
import struct
import sys
from io import BytesIO
from pathlib import Path

import av
import pytest
from PIL import Image

spec = importlib.util.spec_from_file_location(
    "duo_demo", Path(__file__).with_name("server.py")
)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def packet(kind: str, milliseconds: float, value: str) -> module.EngineMediaPacket:
    return {
        "type": kind,
        "response_id": "r",
        "delta": value,
        "sglang": {"media_time": {"t_start_ms": milliseconds}},
    }


def test_delayed_video_keeps_previous_audio_and_drains_tail() -> None:
    assembler = module.MediaAssembler()
    assert assembler.append(packet("response.output_audio.delta", 0, "audio0")) == []
    assert (
        assembler.append(
            packet("response.output_audio.delta", module.BLOCK_MS, "audio1")
        )
        == []
    )
    ready = []
    for index in range(34):
        ready.extend(
            assembler.append(
                packet(
                    "sglang.response.video.delta", index * 1000 / 24, f"frame{index}"
                )
            )
        )
    assert [block["delta"] for block in ready] == ["audio0", "audio1"]
    assert [block["video"]["chunk_index"] for block in ready] == [0, 1]
    assert [[f["data"] for f in block["video"]["frames"]] for block in ready] == [
        [f"frame{i}" for i in range(17)],
        [f"frame{i}" for i in range(17, 34)],
    ]
    assert assembler.blocks == {}


def test_video_arriving_before_audio_waits_for_matching_timestamp() -> None:
    assembler = module.MediaAssembler()
    for index in reversed(range(17)):
        assert (
            assembler.append(
                packet(
                    "sglang.response.video.delta", index * 1000 / 24, f"frame{index}"
                )
            )
            == []
        )
    assert (
        assembler.append(
            packet("response.output_audio.delta", module.BLOCK_MS, "audio1")
        )
        == []
    )
    ready = assembler.append(packet("response.output_audio.delta", 0, "audio0"))
    assert len(ready) == 1
    assert ready[0]["delta"] == "audio0"
    assert [frame["data"] for frame in ready[0]["video"]["frames"]] == [
        f"frame{i}" for i in range(17)
    ]


def test_binary_packet_round_trip_preserves_media_and_json_source() -> None:
    audio = bytes(range(256))
    images = [bytes([index, 255, 0, index + 1]) * 23 for index in range(17)]
    original: module.BrowserMediaPacket = {
        "type": "response.output_audio.delta",
        "response_id": "response",
        "delta": base64.b64encode(audio).decode(),
        "video": {
            "chunk_index": 3,
            "codec": "jpeg",
            "timings": {"submit_to_av_ms": 123.5},
            "frames": [
                {"pts": 51 + index, "data": base64.b64encode(image).decode()}
                for index, image in enumerate(images)
            ],
        },
    }
    encoded = module.encode_browser_packet(original)
    header_size = struct.unpack_from("<I", encoded)[0]
    header = json.loads(encoded[4 : 4 + header_size])
    offset = 4 + header_size
    assert encoded[offset : offset + header["audio_byte_length"]] == audio
    offset += header["audio_byte_length"]
    assert (
        header["video"]["timings"]["submit_to_av_ms"]
        == original["video"]["timings"]["submit_to_av_ms"]
    )
    assert original["video"]["timings"] == {"submit_to_av_ms": 123.5}
    for index, (frame, image) in enumerate(zip(header["video"]["frames"], images)):
        assert frame["pts"] == 51 + index
        assert encoded[offset : offset + frame["byte_length"]] == image
        offset += frame["byte_length"]
    assert offset == len(encoded)
    assert base64.b64decode(original["delta"]) == audio
    assert [
        base64.b64decode(frame["data"]) for frame in original["video"]["frames"]
    ] == images


@pytest.mark.parametrize("incremental", [False, True])
def test_h264_stream_keeps_frame_order_across_packets_and_preserves_audio(
    incremental: bool,
) -> None:
    encoder = module.LiveH264Encoder()
    decoder = av.CodecContext.create("h264", "r")
    decoded_frames: list[av.VideoFrame] = []
    for block in range(2):
        frames = []
        for index in range(17):
            buffer = BytesIO()
            Image.new("RGB", (64, 96), (32 + block * 30 + index, 80, 160)).save(
                buffer, format="JPEG"
            )
            frames.append(
                {
                    "pts": block * 17 + index,
                    "data": base64.b64encode(buffer.getvalue()).decode(),
                }
            )
        original: module.BrowserMediaPacket = {
            "type": "response.output_audio.delta",
            "response_id": "r",
            "delta": base64.b64encode(bytes(range(256))).decode(),
            "video": {
                "chunk_index": block,
                "codec": "jpeg",
                "frames": frames,
                "timings": {"submit_to_av_ms": 600.0},
            },
        }
        encoded_frames = (
            [packet for frame in frames for packet in encoder.encode([frame])]
            if incremental
            else None
        )
        binary = module.encode_browser_packet(original, encoder, encoded_frames)
        size = struct.unpack_from("<I", binary)[0]
        header = json.loads(binary[4 : 4 + size])
        assert header["video"]["codec"] == "h264"
        assert header["video"]["avc_codec"].startswith("avc1.")
        assert header["video"]["frames"][0]["keyframe"] == (block == 0)
        offset = 4 + size
        assert binary[offset : offset + header["audio_byte_length"]] == bytes(
            range(256)
        )
        offset += header["audio_byte_length"]
        for frame in header["video"]["frames"]:
            raw = binary[offset : offset + frame["byte_length"]]
            offset += frame["byte_length"]
            packet = av.Packet(raw)
            packet.pts = frame["pts"]
            decoded_frames.extend(decoder.decode(packet))
        assert offset == len(binary)
        assert all("data" in frame for frame in original["video"]["frames"])
        assert original["video"]["codec"] == "jpeg"
    assert len(decoded_frames) == 34
    assert [frame.pts for frame in decoded_frames] == list(range(34))
    assert all((frame.width, frame.height) == (64, 96) for frame in decoded_frames)


@pytest.mark.parametrize("sample_count", [22666, 22667])
def test_stream_parts_preserve_pcm_frames_and_block_boundary(sample_count: int) -> None:
    audio = bytes(index % 251 for index in range(sample_count * 4))
    frames = [
        {"pts": 51 + index, "data": base64.b64encode(bytes([index]) * 16).decode()}
        for index in range(17)
    ]
    original: module.BrowserMediaPacket = {
        "type": "response.output_audio.delta",
        "response_id": "response",
        "delta": base64.b64encode(audio).decode(),
        "video": {
            "chunk_index": 3,
            "codec": "jpeg",
            "timings": {"submit_to_av_ms": 600.0},
            "frames": frames,
        },
    }
    parts = module.split_browser_packet(original)
    assert [part["video"]["frame_offset"] for part in parts] == [0, 4, 8, 12, 16]
    assert [part["video"]["block_complete"] for part in parts] == [False] * 4 + [True]
    recovered_audio = bytearray()
    recovered_frames = []
    for part in parts:
        packet_bytes = module.encode_browser_packet(part)
        header_size = struct.unpack_from("<I", packet_bytes)[0]
        header = json.loads(packet_bytes[4 : 4 + header_size])
        assert header["video"]["frame_offset"] == part["video"]["frame_offset"]
        assert header["video"]["block_complete"] == part["video"]["block_complete"]
        assert header["audio_byte_length"] % 4 == 0
        offset = 4 + header_size
        recovered_audio.extend(
            packet_bytes[offset : offset + header["audio_byte_length"]]
        )
        recovered_frames.extend(part["video"]["frames"])
    assert recovered_audio == audio
    assert recovered_frames == frames
    parts[0]["video"]["timings"]["gateway_encode_ms"] = 1.0
    assert original["video"]["timings"] == {"submit_to_av_ms": 600.0}
    assert "gateway_encode_ms" not in parts[1]["video"]["timings"]
