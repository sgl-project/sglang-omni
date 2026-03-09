from dataclasses import dataclass, field
from typing import List, Literal, Union

import numpy as np
import torch
from transformers import PreTrainedTokenizerFast

from sglang_omni.models.fishaudio_s2_pro.fish_speech.models.text2semantic.modeling import (
    _get_feat_extract_output_lengths,
)
from sglang_omni.models.fishaudio_s2_pro.fish_speech.tokenizer import (
    AUDIO_EMBED_TOKEN,
    AUDIO_END_TOKEN,
    AUDIO_START_TOKEN,
    IM_END_TOKEN,
    MODALITY_TOKENS,
    SEMANTIC_TOKEN_TEMPLATE,
)


def restore_ndarray(obj, to_tensor: bool = False):
    if isinstance(obj, dict) and "__ndarray__" in obj:
        obj = np.frombuffer(obj["data"], dtype=obj["dtype"]).reshape(obj["shape"])

    if to_tensor and isinstance(obj, np.ndarray):
        obj = torch.from_numpy(obj.copy())

    return obj


@dataclass
class BasePart:
    type: Literal["text", "vq", "audio"] | None = None
    cal_loss: bool = False
    metadata: dict | None = None


@dataclass(kw_only=True)
class VQPart(BasePart):
    type = "vq"
    codes: torch.Tensor

    def __post_init__(self: "VQPart"):
        self.type = "vq"
        self.codes = restore_ndarray(self.codes, to_tensor=True)


@dataclass(kw_only=True)
class TextPart(BasePart):
    type = "text"
    text: str | None = None
    tokens: list[int] | None = None

    def __post_init__(self: "TextPart"):
        self.type = "text"
        if self.text is None and self.tokens is None:
            raise ValueError("Either text or tokens must be provided")


@dataclass(kw_only=True)
class AudioPart(BasePart):
    type = "audio"
    features: torch.Tensor

    # This is the length after downsampling
    length: int | None = None

    def __post_init__(self: "AudioPart"):
        self.type = "audio"
        self.features = restore_ndarray(self.features, to_tensor=True)


@dataclass(kw_only=True)
class EncodedMessage:
    tokens: torch.Tensor
    labels: torch.Tensor
    vq_mask_tokens: torch.Tensor | None = None
    vq_mask_labels: torch.Tensor | None = None
    vq_parts: list[torch.Tensor]
    vq_require_losses: torch.Tensor | None = None
    audio_parts: list[torch.Tensor]
    audio_masks: torch.Tensor | None = None
    audio_lengths: torch.Tensor | None = None
    audio_part_lengths: torch.Tensor | None = None
    audio_feature_masks: torch.Tensor | None = None
    metadata: dict | None = None


@dataclass
class ContentSequence:
    """
    Flexible sequence of content parts that supports interleaved multimodal format.
    Example format: <|interleave|><|speaker:1|> TEXT AUDIO <|im_end|><|speaker:2|> TEXT AUDIO <|im_end|>
    """

    parts: list[BasePart] = field(default_factory=list)
    modality: Literal["text", "voice", "interleave"] | None = None
    metadata: dict | None = None

    def __init__(
        self: "ContentSequence",
        parts: list[BasePart | dict] | None = None,
        modality: Literal["text", "voice", "interleave"] | None = None,
        metadata: dict | None = None,
    ):
        self.modality = modality
        self.metadata = metadata or {}

        fixed_parts = []
        for part in parts or []:
            if isinstance(part, dict):
                if part["type"] == "vq":
                    part = VQPart(**part)
                elif part["type"] == "audio":
                    part = AudioPart(**part)
                elif part["type"] == "text":
                    part = TextPart(**part)
                else:
                    raise ValueError(f"Unsupported part type: {part['type']}")
            fixed_parts.append(part)

        self.parts = fixed_parts

        # If modality is specified, add it at the beginning if it's not already there
        if self.modality and not (
            len(self.parts) > 0
            and isinstance(self.parts[0], dict) is False
            and isinstance(self.parts[0], TextPart)
            and self.parts[0].text is not None
            and self.parts[0].text.startswith(MODALITY_TOKENS[self.modality])
        ):
            modality_token = MODALITY_TOKENS[self.modality]
            self.parts.insert(0, TextPart(text=modality_token))

    def append(
        self: "ContentSequence",
        part_or_parts: Union[BasePart, List[BasePart]],
        add_end: bool = False,
        speaker: Union[str, int] | None = None,
    ):
        """
        Append a part or list of parts to the sequence.

        Args:
            part_or_parts: A single part or list of parts to add
            add_end: Whether to add the IM_END_TOKEN after these parts
            speaker: Optional speaker identifier (name or ID) to add before the parts
        """
        # Convert single part to list
        parts_to_add = (
            [part_or_parts] if not isinstance(part_or_parts, list) else part_or_parts
        )

        # Add speaker token if specified
        if speaker is not None:
            speaker_token = f"<|speaker:{speaker}|>"
            self.parts.append(TextPart(text=speaker_token))

        # Add all the parts
        self.parts.extend(parts_to_add)

        # Add end token if requested
        if add_end:
            self.parts.append(
                TextPart(text=IM_END_TOKEN, cal_loss=self.parts[-1].cal_loss)
            )

    def to_deterministic(
        self: "ContentSequence",
        tokenizer: PreTrainedTokenizerFast,
    ) -> "ContentSequence":
        """
        Convert all TextParts to use tokens instead of text, ensuring deterministic encoding.

        This is useful for RLHF where we need to guarantee that the content sequence
        will always produce the same token sequence when encoded.

        Args:
            tokenizer: The tokenizer to use for encoding text to tokens

        Returns:
            self (for chaining)
        """
        for part in self.parts:
            if isinstance(part, TextPart) and part.text is not None:
                # Encode text to tokens and store
                part.tokens = tokenizer.encode(part.text, add_special_tokens=False)
                part.text = None  # Clear text to use tokens

        return self

    def encode(
        self: "ContentSequence",
        tokenizer: PreTrainedTokenizerFast,
        add_shift: bool = True,
        ignore_loss_tokens: list[str] = [],
        max_length: int | None = None,
    ) -> EncodedMessage:
        """
        Encode the sequence parts into tokens for the model.

        Args:
            tokenizer: The tokenizer to use
            add_shift: Whether to shift tokens for next-token prediction
            ignore_loss_tokens: List of token strings to ignore when calculating loss

        Returns:
            EncodedMessage with tensors ready for the model
        """
        all_tokens = []
        all_labels = []

        # Multi-modal elements
        vq_parts = []
        vq_masks = []
        vq_require_losses = []

        audio_parts = []
        audio_masks = []
        audio_lengths = []
        audio_part_lengths = []

        ignore_loss_token_ids = tokenizer.convert_tokens_to_ids(ignore_loss_tokens)

        for part in self.parts:
            if isinstance(part, TextPart):
                if part.tokens is None:
                    assert part.text is not None
                    tokens = tokenizer.encode(part.text)
                else:
                    tokens = part.tokens

                tokens = torch.tensor(tokens, dtype=torch.int)
            elif isinstance(part, VQPart):
                curr_codes = part.codes.clone().to(torch.int)
                tokens = torch.tensor(
                    tokenizer.convert_tokens_to_ids(
                        [
                            SEMANTIC_TOKEN_TEMPLATE.format(i=i)
                            for i in curr_codes[0].int()
                        ]
                    ),
                    dtype=torch.int,
                )
                vq_parts.append(curr_codes)
                vq_require_losses.append(part.cal_loss)
            elif isinstance(part, AudioPart):
                feature_length = part.length

                if feature_length is None:
                    # Convert mel to 12.5hz, currently hard coded to 8x downsampling
                    feature_length = _get_feat_extract_output_lengths(
                        part.features.shape[1]
                    )

                tokens = torch.tensor(
                    (
                        [tokenizer.convert_tokens_to_ids(AUDIO_START_TOKEN)]
                        + [tokenizer.convert_tokens_to_ids(AUDIO_EMBED_TOKEN)]
                        * feature_length
                        + [tokenizer.convert_tokens_to_ids(AUDIO_END_TOKEN)]
                    ),
                    dtype=torch.int,
                )
                audio_parts.append(part.features)
                audio_lengths.append(feature_length)
                audio_part_lengths.append(part.features.shape[1])
            else:
                raise ValueError(f"Unsupported part type: {type(part)}")

            all_tokens.append(tokens)

            # Set masks for different part types
            if isinstance(part, VQPart):
                vq_masks.append(torch.ones_like(tokens, dtype=torch.bool))
                audio_masks.append(torch.zeros_like(tokens, dtype=torch.bool))
            elif isinstance(part, AudioPart):
                vq_masks.append(torch.zeros_like(tokens, dtype=torch.bool))
                audio_mask = torch.ones_like(tokens, dtype=torch.bool)
                audio_mask[0] = False  # Skip start token
                audio_mask[-1] = False  # Skip end token
                audio_masks.append(audio_mask)
            else:
                vq_masks.append(torch.zeros_like(tokens, dtype=torch.bool))
                audio_masks.append(torch.zeros_like(tokens, dtype=torch.bool))

            # Set labels based on whether we want to calculate loss for this part
            if part.cal_loss and not isinstance(part, AudioPart):
                all_labels.append(tokens.clone())
            else:
                all_labels.append(torch.full_like(tokens, -100))

        # Concatenate all tensors
        tokens = torch.cat(all_tokens, dim=0)
        labels = torch.cat(all_labels, dim=0)
        vq_masks = torch.cat(vq_masks, dim=0)
        audio_masks = torch.cat(audio_masks, dim=0)
        vq_require_losses = torch.tensor(vq_require_losses, dtype=torch.bool)
        audio_lengths = torch.tensor(
            audio_lengths, dtype=torch.int, device=tokens.device
        )
        audio_part_lengths = torch.tensor(
            audio_part_lengths, dtype=torch.int, device=tokens.device
        )

        # Apply shift if needed for next-token prediction
        vq_mask_tokens = vq_masks
        vq_mask_labels = vq_masks

        if add_shift:
            tokens = tokens[:-1]
            labels = labels[1:]

            # if the last token is a vq_token, we need to remove the corresponding vq_part
            if vq_masks[-1]:
                vq_parts[-1] = vq_parts[-1][:, :-1]
                if vq_parts[-1].shape[1] == 0:
                    vq_parts.pop()
                    vq_require_losses = vq_require_losses[:-1]

            vq_masks = vq_masks[:-1]
            vq_mask_tokens = vq_mask_tokens[:-1]
            vq_mask_labels = vq_mask_labels[1:]

            # if the last token is an audio_token, we need to remove the corresponding audio_part
            assert not audio_masks[-1], "Last token cannot be audio token when shifting"

            audio_masks = audio_masks[:-1]

        # Ignore specified tokens
        for i in ignore_loss_token_ids:
            assert i != -100 and i is not None
            labels[labels == i] = -100

        assert tokens.dtype in [
            torch.int,
            torch.long,
        ], f"Invalid dtype: {tokens.dtype}"

        audio_feature_masks = torch.full((audio_lengths.sum(),), True, dtype=torch.bool)
        if max_length is not None and tokens.shape[0] > max_length:
            tokens = tokens[:max_length]
            labels = labels[:max_length]
            vq_mask_tokens = vq_mask_tokens[:max_length]
            vq_mask_labels = vq_mask_labels[:max_length]
            audio_masks = audio_masks[:max_length]
            count_remaining_audio_tokens = audio_masks.sum().item()

            # No longer pick audio features beyond the max length
            audio_feature_masks[count_remaining_audio_tokens:] = False

            # Sum audio lengths
            count_audio_lengths = 0
            for i in range(len(audio_lengths)):
                feat_len = audio_lengths[i].item()
                count_audio_lengths += feat_len

                # Remove overlong audios
                if count_audio_lengths >= count_remaining_audio_tokens:
                    audio_lengths = audio_lengths[: i + 1]
                    audio_part_lengths = audio_part_lengths[: i + 1]
                    audio_parts = audio_parts[: i + 1]
                    break

        return EncodedMessage(
            tokens=tokens,
            labels=labels,
            vq_parts=vq_parts,
            vq_mask_tokens=vq_mask_tokens,
            vq_mask_labels=vq_mask_labels,
            vq_require_losses=vq_require_losses,
            audio_parts=audio_parts,
            audio_masks=audio_masks,
            audio_lengths=audio_lengths,
            audio_part_lengths=audio_part_lengths,
            audio_feature_masks=audio_feature_masks,
            metadata=self.metadata,
        )

    def visualize(
        self: "ContentSequence",
        tokenizer: PreTrainedTokenizerFast,
        ignore_loss_tokens: list[str] = [],
        merge_semantic_tokens: bool = False,
        merge_audio_tokens: bool = False,
        use_color: bool = True,
    ):
        """
        Visualize the encoded sequence with color-coded tokens.
        Blue/cyan tokens contribute to loss, green tokens do not.
        """
        encoded = self.encode(
            tokenizer, add_shift=False, ignore_loss_tokens=ignore_loss_tokens
        )

        # Colors for alternating tokens
        colors = {
            "blue": "\033[94m",  # Light blue
            "cyan": "\033[96m",  # Cyan
            "green": "\033[92m",  # Light green
            "dark_green": "\033[32m",  # Dark green
        }
        blue_idx = 0
        green_idx = 0

        def print_in_blue(x):
            if not use_color:
                print(x, end="")
                return

            nonlocal blue_idx
            color = colors["blue"] if blue_idx % 2 == 0 else colors["cyan"]
            print(f"{color}{x}\033[0m", end="")
            blue_idx += 1

        def print_in_green(x):
            if not use_color:
                print(x, end="")
                return

            nonlocal green_idx
            color = colors["green"] if green_idx % 2 == 0 else colors["dark_green"]
            print(f"{color}{x}\033[0m", end="")
            green_idx += 1

        def print_semantic_token(require_loss, count):
            val = f"[<|semantic|>x{count}]"
            if require_loss:
                print_in_blue(val)
            else:
                print_in_green(val)

        def print_audio_token(count):
            val = f"[<|audio_pad|>x{count}]"
            print_in_green(val)

        count_semantic_tokens = 0
        semantic_require_loss = None
        count_audio_tokens = 0

        # Buffer for grouping continuous text tokens (for proper UTF-8 decoding)
        text_token_buffer = []
        text_token_require_loss = None

        def flush_text_buffer():
            nonlocal text_token_buffer, text_token_require_loss
            if text_token_buffer:
                val = tokenizer.decode(text_token_buffer)
                if text_token_require_loss:
                    print_in_blue(val)
                else:
                    print_in_green(val)
                text_token_buffer = []
                text_token_require_loss = None

        # Get all special tokens, found min & max IDs for semantic tokens
        all_semantic_ids = [
            k
            for k, v in tokenizer.added_tokens_decoder.items()
            if "semantic" in v.content
        ]
        min_semantic_id = min(all_semantic_ids)
        max_semantic_id = max(all_semantic_ids)
        audio_embed_id = tokenizer.convert_tokens_to_ids(AUDIO_EMBED_TOKEN)

        for tok, lab in zip(encoded.tokens, encoded.labels):
            token_id = int(tok.item())
            require_loss = lab != -100

            if merge_semantic_tokens:
                if min_semantic_id <= token_id <= max_semantic_id and (
                    semantic_require_loss is None
                    or semantic_require_loss == require_loss
                ):
                    flush_text_buffer()
                    count_semantic_tokens += 1
                    semantic_require_loss = require_loss
                    continue
                elif count_semantic_tokens > 0:
                    print_semantic_token(semantic_require_loss, count_semantic_tokens)
                    count_semantic_tokens = 0
                    semantic_require_loss = None

            if merge_audio_tokens:
                if token_id == audio_embed_id:
                    flush_text_buffer()
                    count_audio_tokens += 1
                    continue
                elif count_audio_tokens > 0:
                    print_audio_token(count_audio_tokens)
                    count_audio_tokens = 0

            # Group continuous text tokens with the same loss requirement
            if text_token_require_loss is None:
                text_token_require_loss = require_loss

            if require_loss == text_token_require_loss:
                text_token_buffer.append(token_id)
            else:
                flush_text_buffer()
                text_token_buffer.append(token_id)
                text_token_require_loss = require_loss

        # Flush remaining buffers
        flush_text_buffer()

        if merge_semantic_tokens and count_semantic_tokens > 0:
            print_semantic_token(semantic_require_loss, count_semantic_tokens)

        print()


if __name__ == "__main__":
    # Example of using the new ContentSequence format
    from transformers import PreTrainedTokenizerFast

    tokenizer = PreTrainedTokenizerFast.from_pretrained("checkpoints/agent-0.6b-debug")

    # Create a conversation with interleaved format
    seq = ContentSequence(modality="interleave")

    # Add first speaker parts
    seq.append(TextPart(text="Hello, how are you?", cal_loss=False), speaker=1)
    seq.append(
        VQPart(codes=torch.randint(0, 100, (5, 10)), cal_loss=True), add_end=True
    )

    # Add second speaker parts
    seq.append(TextPart(text="I'm fine, thank you.", cal_loss=True), speaker=2)
    seq.append(
        VQPart(codes=torch.randint(0, 100, (3, 10)), cal_loss=True), add_end=True
    )

    # Add another exchange with named speakers
    # seq.append(
    #     TextPart(text="What's the weather like?", cal_loss=False),
    #     speaker="alice",
    #     add_end=True,
    # )
    # seq.append(
    #     TextPart(text="It's sunny today.", cal_loss=True), speaker="bob", add_end=True
    # )

    # Visualize the conversation
    seq.visualize(tokenizer)

    tokenized = seq.encode(tokenizer)
    print(tokenized.tokens.tolist(), tokenized.labels.tolist())
    print(tokenized.vq_mask_tokens.tolist())
    print(tokenized.vq_mask_labels.tolist())
    print(tokenized.vq_parts)
    print(tokenized.vq_require_losses.tolist())
