(function (global) {
  "use strict";

  class RealtimePlaybackController {
    constructor(options = {}) {
      this.sampleRate = options.sampleRate || 24000;
      this.createAudioContext =
        options.createAudioContext ||
        (() => new (global.AudioContext || global.webkitAudioContext)());
      this.decodeBase64 =
        options.decodeBase64 ||
        ((encoded) => {
          const binary = global.atob(encoded);
          return Uint8Array.from(binary, (value) => value.charCodeAt(0));
        });
      this.context = null;
      this.nextPlaybackTime = 0;
      this.activeResponseId = null;
      this.playbackResponseId = null;
      this.cancelledResponseIds = new Set();
      this.responseMetadata = new Map();
      this.sources = new Set();
    }

    ensureContext() {
      if (!this.context || this.context.state === "closed") {
        this.context = this.createAudioContext();
        this.nextPlaybackTime = 0;
      }
      if (this.context.state === "suspended") this.context.resume();
      return this.context;
    }

    beginResponse(responseId) {
      this.activeResponseId = responseId || null;
      if (responseId) {
        this.responseMetadata.set(responseId, {
          itemId: null,
          playbackStartTime: null,
          finished: false,
        });
        this.boundResponseMetadata();
      }
    }

    finishResponse(responseId, status) {
      const metadata = this.responseMetadata.get(responseId);
      if (metadata) metadata.finished = true;
      if (status !== "completed") {
        this.rejectResponse(responseId);
      }
      if (responseId && responseId === this.activeResponseId) {
        this.activeResponseId = null;
      }
    }

    rejectResponse(responseId) {
      if (!responseId) return;
      this.cancelledResponseIds.add(responseId);
      if (responseId === this.activeResponseId) this.activeResponseId = null;
      if (responseId === this.playbackResponseId) this.flush();
      this.responseMetadata.delete(responseId);
      while (this.cancelledResponseIds.size > 64) {
        this.cancelledResponseIds.delete(
          this.cancelledResponseIds.values().next().value,
        );
      }
    }

    interrupt() {
      const responseIds = new Set(
        [this.activeResponseId, this.playbackResponseId].filter(Boolean),
      );
      const records = [];
      responseIds.forEach((responseId) => {
        const metadata = this.responseMetadata.get(responseId);
        const playbackStartTime = metadata && metadata.playbackStartTime;
        records.push({
          responseId,
          itemId: (metadata && metadata.itemId) || null,
          audioEndMs:
            this.context &&
            playbackStartTime !== null &&
            playbackStartTime !== undefined
              ? Math.max(
                  0,
                  Math.floor(
                    (this.context.currentTime - playbackStartTime) * 1000,
                  ),
                )
              : 0,
        });
        if (!metadata || !metadata.finished) {
          this.cancelledResponseIds.add(responseId);
        }
        this.responseMetadata.delete(responseId);
      });
      while (this.cancelledResponseIds.size > 64) {
        this.cancelledResponseIds.delete(
          this.cancelledResponseIds.values().next().value,
        );
      }
      this.activeResponseId = null;
      this.flush();
      return records;
    }

    queueAudioDelta(encoded, responseId, itemId) {
      if (!responseId || this.cancelledResponseIds.has(responseId)) return false;
      const bytes = this.decodeBase64(encoded);
      if (bytes.byteLength % 2 !== 0) {
        throw new Error("Received an odd-length PCM16 audio delta");
      }
      if (this.cancelledResponseIds.has(responseId)) return false;

      const context = this.ensureContext();
      const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
      const samples = new Float32Array(bytes.byteLength / 2);
      for (let index = 0; index < samples.length; index++) {
        samples[index] = view.getInt16(index * 2, true) / 0x8000;
      }
      const buffer = context.createBuffer(1, samples.length, this.sampleRate);
      buffer.copyToChannel(samples, 0);
      const source = context.createBufferSource();
      source.buffer = buffer;
      source.connect(context.destination);
      source.onended = () => {
        this.sources.delete(source);
        if (
          this.sources.size === 0 &&
          this.playbackResponseId === responseId
        ) {
          this.playbackResponseId = null;
        }
      };

      const startAt = Math.max(
        context.currentTime + 0.02,
        this.nextPlaybackTime,
      );
      const metadata = this.responseMetadata.get(responseId) || {
        itemId: null,
        playbackStartTime: null,
        finished: false,
      };
      if (itemId) metadata.itemId = itemId;
      if (metadata.playbackStartTime === null) {
        metadata.playbackStartTime = startAt;
      }
      this.responseMetadata.set(responseId, metadata);
      this.boundResponseMetadata();
      this.playbackResponseId = responseId;
      this.sources.add(source);
      source.start(startAt);
      this.nextPlaybackTime = startAt + buffer.duration;
      return true;
    }

    flush(closeContext = false) {
      this.sources.forEach((source) => {
        try {
          source.stop();
        } catch (_) {}
      });
      this.sources.clear();
      if (closeContext && this.context) {
        this.context.close();
        this.context = null;
      }
      this.playbackResponseId = null;
      this.nextPlaybackTime = this.context ? this.context.currentTime : 0;
    }

    close() {
      this.flush(true);
      this.activeResponseId = null;
      this.cancelledResponseIds.clear();
      this.responseMetadata.clear();
    }

    boundResponseMetadata() {
      while (this.responseMetadata.size > 64) {
        const responseId = this.responseMetadata.keys().next().value;
        this.responseMetadata.delete(responseId);
        this.cancelledResponseIds.delete(responseId);
      }
    }
  }

  class RealtimeTurnTracker {
    constructor() {
      this.pendingItemIds = [];
      this.respondingItemId = null;
      this.activeResponseId = null;
      this.responseItems = new Map();
      this.staleResponseIds = new Set();
      this.pendingResponseInterrupted = false;
    }

    commit(itemId) {
      if (itemId) this.pendingItemIds.push(itemId);
    }

    hasPendingResponse() {
      return this.pendingItemIds.length > 0;
    }

    markPendingInterrupted() {
      if (this.hasPendingResponse()) this.pendingResponseInterrupted = true;
    }

    clearPendingInterruption() {
      this.pendingResponseInterrupted = false;
    }

    beginResponse(responseId) {
      const itemId = this.pendingItemIds.shift() || null;
      const interrupted = this.pendingResponseInterrupted;
      this.pendingResponseInterrupted = false;
      if (!responseId || !itemId) {
        this.markResponseStale(responseId);
        return { itemId, interrupted: true };
      }
      this.responseItems.set(responseId, { itemId, finished: false });
      if (interrupted) {
        this.markResponseStale(responseId);
      } else {
        this.activeResponseId = responseId;
        this.respondingItemId = itemId;
      }
      while (this.responseItems.size > 64) {
        const staleResponseId = this.responseItems.keys().next().value;
        this.responseItems.delete(staleResponseId);
        this.staleResponseIds.delete(staleResponseId);
      }
      return { itemId, interrupted };
    }

    interruptResponse(responseId) {
      if (!responseId) return null;
      const response = this.responseItems.get(responseId);
      if (response && !response.finished) {
        this.markResponseStale(responseId);
      }
      const itemId = response ? response.itemId : null;
      if (responseId === this.activeResponseId) {
        this.activeResponseId = null;
        this.respondingItemId = null;
      }
      return itemId;
    }

    ownsResponse(responseId) {
      return Boolean(responseId) && responseId === this.activeResponseId;
    }

    finishResponse(responseId, reason = null) {
      const response = this.responseItems.get(responseId);
      const itemId = response ? response.itemId : null;
      const interrupted =
        this.staleResponseIds.has(responseId) || reason === "turn_detected";
      if (response) response.finished = true;
      if (responseId === this.activeResponseId) {
        this.activeResponseId = null;
        this.respondingItemId = null;
      }
      this.staleResponseIds.delete(responseId);
      return { itemId, interrupted };
    }

    markResponseStale(responseId) {
      if (!responseId) return;
      this.staleResponseIds.add(responseId);
      while (this.staleResponseIds.size > 64) {
        this.staleResponseIds.delete(
          this.staleResponseIds.values().next().value,
        );
      }
    }

    clear() {
      this.pendingItemIds.length = 0;
      this.respondingItemId = null;
      this.activeResponseId = null;
      this.responseItems.clear();
      this.staleResponseIds.clear();
      this.pendingResponseInterrupted = false;
    }
  }

  global.RealtimePlaybackController = RealtimePlaybackController;
  global.RealtimeTurnTracker = RealtimeTurnTracker;
  if (typeof module !== "undefined" && module.exports) {
    module.exports = { RealtimePlaybackController, RealtimeTurnTracker };
  }
})(typeof window !== "undefined" ? window : globalThis);
