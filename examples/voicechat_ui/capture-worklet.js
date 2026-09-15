// SPDX-License-Identifier: Apache-2.0
import { PCMFramer } from "./audio.mjs";
class VoiceChatCapture extends AudioWorkletProcessor {
    constructor(options) {
        super();
        this.muted = false;
        this.framer = new PCMFramer(
            sampleRate,
            options.processorOptions.targetRate,
            options.processorOptions.frameSamples,
        );
        this.port.onmessage = ({ data }) => {
            this.muted = Boolean(data.muted);
        };
    }
    process(inputs) {
        const channel = inputs[0]?.[0];
        if (!channel) return true;
        let power = 0;
        for (const x of channel) power += x * x;
        // Copy when muted; never alter a shared input buffer supplied by the browser.
        this.framer.push(
            this.muted ? new Float32Array(channel.length) : channel,
            (frame) => {
                this.port.postMessage(
                    {
                        pcm: frame,
                        level: this.muted
                            ? 0
                            : Math.sqrt(power / channel.length),
                    },
                    [frame.buffer],
                );
            },
        );
        return true;
    }
}
registerProcessor("voicechat-capture", VoiceChatCapture);
