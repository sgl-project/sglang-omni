import argparse
import logging
from dataclasses import dataclass, fields
from typing import List, Optional

from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)

OMNI_DISABLED_OR_NOT_IMPLEMENTED_SERVER_ARGS = [
    "enable_mixed_chunk",
    "enable_dynamic_chunking",
]


@dataclass
class SGLangBackendArgs:
    disabled_args: Optional[List[str]] = None

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> None:

        # SGLang arguments
        ServerArgs.add_cli_args(parser)

        # sglang-omni extra arguments
        parser.add_argument(
            "--disabled-args",
            type=str,
            nargs="+",
            default=None,
            help=(
                "List of argument names that are not allowed to be passed. "
                "Supports only dest names (e.g. base_gpu_id)."
            ),
        )

    @staticmethod
    def _remove_disabled_args(args: argparse.Namespace) -> argparse.Namespace:
        if not args.disabled_args:
            return args

        disaged_args_list = set(
            args.disabled_args + OMNI_DISABLED_OR_NOT_IMPLEMENTED_SERVER_ARGS
        )
        for raw_arg in disaged_args_list:
            arg = raw_arg.lstrip("-").replace("-", "_")
            if hasattr(args, arg):
                delattr(args, arg)
                logger.warning(f"SGLang Argument `{raw_arg}` is disabled.")

        return args

    @staticmethod
    def _strip_backend_args(args: argparse.Namespace) -> argparse.Namespace:
        backend_arg_names = {f.name for f in fields(SGLangBackendArgs)}
        stripped = {
            key: value
            for key, value in vars(args).items()
            if key not in backend_arg_names
        }
        return argparse.Namespace(**stripped)


def prepare_sgl_server_args(argv: List[str]) -> ServerArgs:
    parser = argparse.ArgumentParser()
    SGLangBackendArgs.add_cli_args(parser)

    raw_args = parser.parse_args(argv)
    SGLangBackendArgs._remove_disabled_args(raw_args)
    cli_server_args = SGLangBackendArgs._strip_backend_args(raw_args)

    server_args = ServerArgs.from_cli_args(cli_server_args)

    return server_args
