'use strict';
// Stereo WSOLA. Both channels use the same alignment and overlap weights.
function catchupAudio(input,rate,ctx){
  if(rate<=1.001)return input;
  const size=Math.round(input.length/rate),windowSize=1024,overlap=256,hop=windowSize-overlap;
  if(size<windowSize)return input;
  const output=ctx.createBuffer(input.numberOfChannels,size,input.sampleRate);
  const source=Array.from({length:input.numberOfChannels},(_,ch)=>input.getChannelData(ch));
  const dest=source.map((_,ch)=>output.getChannelData(ch));
  for(let ch=0;ch<dest.length;ch++)dest[ch].set(source[ch].subarray(0,windowSize));
  for(let pos=hop;pos<size;pos+=hop){
    const count=Math.min(windowSize,size-pos),blend=Math.min(overlap,count);
    const expected=Math.round(pos*rate),limit=input.length-count;
    let best=Math.min(expected,limit),bestScore=-Infinity;
    const first=Math.max(0,best-512),last=Math.min(limit,best+512);
    let referenceEnergy=0;
    for(let j=0;j<blend;j+=4)referenceEnergy+=dest[0][pos+j]**2;
    function score(candidate){
      let cross=0,energy=0;
      for(let j=0;j<blend;j+=4){const x=source[0][candidate+j];cross+=dest[0][pos+j]*x;energy+=x*x}
      return cross/Math.sqrt(referenceEnergy*energy+1e-15);
    }
    if(referenceEnergy>1e-10){
      for(let candidate=first;candidate<=last;candidate+=4){const value=score(candidate);if(value>bestScore){bestScore=value;best=candidate}}
      const coarse=best;
      for(let candidate=Math.max(first,coarse-3);candidate<=Math.min(last,coarse+3);candidate++){const value=score(candidate);if(value>bestScore){bestScore=value;best=candidate}}
    }
    if(count<windowSize)best=limit;
    for(let ch=0;ch<dest.length;ch++){
      for(let j=0;j<blend;j++){const mix=(1-Math.cos(Math.PI*j/blend))/2;dest[ch][pos+j]=dest[ch][pos+j]*(1-mix)+source[ch][best+j]*mix}
      dest[ch].set(source[ch].subarray(best+blend,best+count),pos+blend);
    }
  }
  return output;
}
