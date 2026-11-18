class MicCapture extends AudioWorkletProcessor {
  constructor(){super();this.sum=0;this.count=0;this.phase=0;this.batch=new Float32Array(256);this.index=0}
  process(inputs){const ch=inputs[0]?.[0];if(ch)for(const x of ch){this.sum+=x;this.count++;this.phase+=16000;if(this.phase>=sampleRate){this.phase-=sampleRate;this.batch[this.index++]=this.sum/this.count;this.sum=0;this.count=0;if(this.index===256){this.port.postMessage(this.batch,[this.batch.buffer]);this.batch=new Float32Array(256);this.index=0}}}return true}
}
registerProcessor('mic-capture',MicCapture);
