import argparse

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from sglang_omni.realtime.backend import MockResponseBackend
from sglang_omni.serve.webrtc_api import create_realtime_router


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mock realtime API for browser smoke tests"
    )
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model-name", type=str, default="mock-realtime")
    parser.add_argument(
        "--response-text",
        type=str,
        default="Mock backend streaming a test tone.",
    )
    parser.add_argument(
        "--audio-mode",
        type=str,
        choices=("tone", "echo"),
        default="tone",
    )
    parser.add_argument("--dump-audio-dir", type=str, default=None)
    parser.add_argument("--sample-rate", type=int, default=24000)
    parser.add_argument("--chunk-duration", type=float, default=0.24)
    parser.add_argument("--chunk-delay", type=float, default=0.08)
    parser.add_argument("--total-duration", type=float, default=1.2)
    parser.add_argument("--tone-frequency", type=float, default=660.0)
    return parser.parse_args()


def create_app(args: argparse.Namespace) -> FastAPI:
    app = FastAPI(title="sglang-omni-realtime-mock", version="0.1.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    def backend_factory(model_name: str, max_new_tokens: int, output_text: bool):
        del max_new_tokens
        output_modalities = ("text", "audio") if output_text else ("audio",)
        return MockResponseBackend(
            model=model_name,
            output_modalities=output_modalities,
            response_text=args.response_text,
            audio_mode=args.audio_mode,
            dump_audio_dir=args.dump_audio_dir,
            sample_rate=args.sample_rate,
            chunk_duration_s=args.chunk_duration,
            inter_chunk_delay_s=args.chunk_delay,
            total_duration_s=args.total_duration,
            tone_hz=args.tone_frequency,
        )

    app.include_router(
        create_realtime_router(
            model_name=args.model_name,
            backend_factory=backend_factory,
            audio_debug_dump_dir=args.dump_audio_dir,
        )
    )

    @app.get("/health")
    async def health() -> JSONResponse:
        return JSONResponse(
            {
                "status": "healthy",
                "running": True,
                "backend": "mock-realtime",
                "model": args.model_name,
            }
        )

    return app


def main() -> None:
    args = parse_args()
    uvicorn.run(create_app(args), host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
