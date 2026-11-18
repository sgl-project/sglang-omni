'use strict';
function nextInputBlock(queue,accelerate,ctx,finished=false){
  const available=queue.reduce((n,b)=>n+b.frames.length,0);
  if(!available||available<17&&!finished)return null;
  const uploaded=!!queue[0].uploaded;
  const count=Math.min(available,!uploaded&&accelerate&&available>=19?19:17);
  const frames=[],parts=[],capturedAt=queue[0].capturedAt;
  let needed=count,speech=false;
  while(needed){
    const b=queue[0],take=Math.min(needed,b.frames.length);
    const stop=Math.round(b.samples.length*take/b.frames.length);
    frames.push(...b.frames.slice(0,take));parts.push(b.samples.slice(0,stop));speech||=b.speech;
    b.frames=b.frames.slice(take);b.samples=b.samples.slice(stop);b.capturedAt+=take*1000/24;
    if(!b.frames.length)queue.shift();
    needed-=take;
  }
  const samples=new Float32Array(parts.reduce((n,p)=>n+p.length,0));
  let offset=0;for(const p of parts){samples.set(p,offset);offset+=p.length}
  let pcm=samples;
  if(count>17){
    const input=ctx.createBuffer(1,samples.length,16000);input.getChannelData(0).set(samples);
    pcm=catchupAudio(input,count/17,ctx).getChannelData(0);
  }else if(count<17){
    pcm=new Float32Array(Math.round(17*16000/24));pcm.set(samples);
  }
  const selected=Array.from({length:17},(_,i)=>frames[count>17?Math.round(i*(count-1)/16):Math.min(i,count-1)]);
  return {frames:selected,samples:pcm,speech,capturedAt,uploaded,accelerated:count>17,consumedFrames:count};
}
