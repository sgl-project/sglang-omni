"""Generation inputs."""

import torch.utils.checkpoint
from transformers import PreTrainedTokenizerBase

from sglang_omni.models.sensenova_u1.neo_unify.attention_backend import (
    create_block_causal_mask,
)
from sglang_omni.models.sensenova_u1.neo_unify.conversation import get_conv_template
from sglang_omni.models.sensenova_u1.neo_unify.modeling_neo_chat import NEOChatModel
from sglang_omni.models.sensenova_u1.neo_unify.modeling_neo_vit import (
    build_abs_positions_from_grid_hw,
)
from sglang_omni.models.sensenova_u1.neo_unify.utils import SYSTEM_MESSAGE_FOR_GEN
from sglang_omni.pipeline.umm import UMMRequestError

# Modified for SGLang; see this directory's README.md for upstream source.


def patchify(
    images: torch.Tensor,
    patch_size: int,
    channel_first: bool = False,
) -> torch.Tensor:
    """images: (N, 3, H, W)"""
    h, w = images.shape[2] // patch_size, images.shape[3] // patch_size
    x = images.reshape(shape=(images.shape[0], 3, h, patch_size, w, patch_size))

    if channel_first:
        x = torch.einsum("nchpwq->nhwcpq", x)
    else:
        x = torch.einsum("nchpwq->nhwpqc", x)

    x = x.reshape(shape=(images.shape[0], h * w, patch_size**2 * 3))
    return x


def unpatchify(
    x: torch.Tensor,
    patch_size: int,
    h: int | None = None,
    w: int | None = None,
) -> torch.Tensor:
    """x: (N, L, patch_size**2 *3)"""
    if h is None or w is None:
        h = w = int(x.shape[1] ** 0.5)
    else:
        h = h // patch_size
        w = w // patch_size
    x = x.reshape(shape=(x.shape[0], h, w, patch_size, patch_size, 3))
    x = torch.einsum("nhwpqc->nchpwq", x)
    images = x.reshape(shape=(x.shape[0], 3, h * patch_size, w * patch_size))
    return images


def build_t2i_query(
    model: NEOChatModel,
    prompt_text: str,
    system_message: str | None = None,
    append_text: str | None = None,
) -> str:
    template = get_conv_template(model.template)
    template.system_message = (
        model.system_message if system_message is None else system_message
    )
    template.append_message(template.roles[0], prompt_text)
    template.append_message(template.roles[1], None)
    if append_text is not None:
        return template.get_prompt() + append_text
    else:
        pass
    return template.get_prompt()


def build_t2i_queries(
    model: NEOChatModel, prompts: list[str], guidance_scale: float
) -> list[list[str]]:
    queries = [
        [
            build_t2i_query(
                model,
                prompt,
                system_message=SYSTEM_MESSAGE_FOR_GEN,
                append_text="<think>\n\n</think>\n\n<img>",
            )
            for prompt in prompts
        ]
    ]
    if guidance_scale > 1:
        queries.append([build_t2i_query(model, "", append_text="<img>")])
    else:
        pass
    return queries


def encode_generation_query(
    model: NEOChatModel, tokenizer: PreTrainedTokenizerBase, query: str
) -> torch.Tensor:
    token_ids = tokenizer(query, return_tensors="pt")["input_ids"][0]
    # note (Wes): Generated image patches use the next temporal position after the prefix.
    max_prefix_tokens = min(
        tokenizer.model_max_length, model.config.llm_config.max_position_embeddings - 1
    )
    if token_ids.numel() > max_prefix_tokens:
        raise UMMRequestError(
            f"SenseNova-U1 input contains {token_ids.numel()} tokens after template "
            f"and reference expansion; maximum is {max_prefix_tokens} tokens"
        )
    else:
        return token_ids


def build_t2i_text_inputs(
    model: NEOChatModel, tokenizer: PreTrainedTokenizerBase, query: str | list[str]
) -> tuple[
    torch.Tensor, torch.Tensor, dict[str, torch.Tensor], torch.Tensor, torch.Tensor
]:
    queries = [query] if isinstance(query, str) else query
    if not queries:
        raise ValueError("query batch must not be empty")
    else:
        pass
    encoded = [encode_generation_query(model, tokenizer, query) for query in queries]
    lengths = [item.shape[0] for item in encoded]
    prefix_lengths = torch.tensor(lengths, dtype=torch.long, device=model.device)
    max_length = max(lengths)
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = getattr(tokenizer, "eos_token_id", None)
    else:
        pass
    if pad_token_id is None:
        pad_token_id = 0
    else:
        pass

    input_ids = torch.full(
        (len(encoded), max_length),
        int(pad_token_id),
        dtype=encoded[0].dtype,
        device=model.device,
    )
    key_valid_mask = torch.zeros(
        (len(encoded), max_length), dtype=torch.bool, device=model.device
    )
    for batch_index, item in enumerate(encoded):
        item = item.to(model.device)
        item_length = item.shape[0]
        input_ids[batch_index, :item_length] = item
        key_valid_mask[batch_index, :item_length] = True

    t_idx = torch.arange(max_length, dtype=torch.long, device=input_ids.device).expand(
        len(encoded), -1
    )
    h_idx = torch.zeros_like(t_idx)
    w_idx = torch.zeros_like(t_idx)
    batched_indexes = torch.stack([t_idx, h_idx, w_idx], dim=1)
    indexes = batched_indexes[0] if len(encoded) == 1 else batched_indexes

    attention_mask = {"full_attention": create_block_causal_mask(t_idx, key_valid_mask)}
    return input_ids, indexes, attention_mask, key_valid_mask, prefix_lengths


def build_t2i_image_indexes(
    token_h: int,
    token_w: int,
    text_len: int | torch.Tensor,
    device: torch.device | None,
) -> torch.Tensor:
    if isinstance(text_len, torch.Tensor):
        text_len = text_len.to(device=device, dtype=torch.long).reshape(-1)
        batch_size = text_len.shape[0]
        image_len = token_h * token_w
        t_image = text_len[:, None].expand(batch_size, image_len)
        idx = torch.arange(image_len, device=device, dtype=torch.long)
        h_image = (idx // token_w).expand(batch_size, -1)
        w_image = (idx % token_w).expand(batch_size, -1)
        return torch.stack([t_image, h_image, w_image], dim=1)
    else:
        pass

    t_image = torch.full(
        (token_h * token_w,), text_len, dtype=torch.long, device=device
    )
    idx = torch.arange(token_h * token_w, device=device, dtype=torch.long)
    h_image = idx // token_w
    w_image = idx % token_w
    return torch.stack([t_image, h_image, w_image], dim=0)


def build_it2i_inputs(
    model: NEOChatModel,
    tokenizer: PreTrainedTokenizerBase,
    query: str,
    pixel_values: torch.Tensor | None = None,
    grid_hw: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    input_ids = (
        encode_generation_query(model, tokenizer, query).unsqueeze(0).to(model.device)
    )

    indexes = get_thw_indexes(model, input_ids[0], grid_hw)

    attention_mask = {"full_attention": create_block_causal_mask(indexes[0])}

    input_embeds = model.language_model.get_input_embeddings()(input_ids)
    B, N, C = input_embeds.shape
    if pixel_values is not None:
        vit_embeds = model.extract_feature(pixel_values, grid_hw=grid_hw)
        input_embeds = input_embeds.reshape(B * N, C)
        input_ids = input_ids.reshape(B * N)
        selected = input_ids == model.img_context_token_id
        assert selected.sum() != 0
        input_embeds[selected] = vit_embeds.reshape(-1, C).to(input_embeds.device)
        input_embeds = input_embeds.reshape(B, N, C)
    else:
        pass

    return input_embeds, indexes, attention_mask


def get_thw_indexes(
    model: NEOChatModel, input_ids: torch.Tensor, grid_hw: torch.Tensor | None = None
) -> torch.Tensor:
    img_start_shift = torch.cat(
        [
            torch.zeros(1, dtype=torch.long).to(input_ids.device),
            (input_ids == model.img_start_token_id).long(),
        ],
        dim=0,
    )[:-1]
    not_img_token = (input_ids != model.img_context_token_id).long()
    t_indexes = (img_start_shift + not_img_token).cumsum(0) - 1
    h_indexes = torch.zeros_like(t_indexes).to(t_indexes.device)
    w_indexes = torch.zeros_like(t_indexes).to(t_indexes.device)

    if grid_hw is not None:
        selected = input_ids == model.img_context_token_id
        if selected.long().sum() > 0:
            abs_pos_w, abs_pos_h = build_abs_positions_from_grid_hw(
                grid_hw // int(1 / model.downsample_ratio), device=t_indexes.device
            )
            h_indexes[selected] = abs_pos_h.to(t_indexes.device, t_indexes.dtype)
            w_indexes[selected] = abs_pos_w.to(t_indexes.device, t_indexes.dtype)
        else:
            pass
    else:
        pass
    return torch.stack([t_indexes, h_indexes, w_indexes], dim=0)
