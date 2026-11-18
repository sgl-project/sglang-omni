'use strict';
async function h264Supported(){
  if(typeof VideoDecoder==='undefined')return false;
  return (await VideoDecoder.isConfigSupported({codec:'avc1.4d401f'})).supported;
}
async function decodeVideoFrames(s,video){
  if(video.codec!=='h264'){
    return Promise.all(video.frames.map(f=>{
      const img=new Image(),url=f.encodedBytes?URL.createObjectURL(new Blob([f.encodedBytes],{type:'image/jpeg'})):null;
      img.src=url||'data:image/jpeg;base64,'+f.data;
      return img.decode().then(()=>img).finally(()=>{if(url)URL.revokeObjectURL(url)});
    }));
  }
  if(!s.decoder){
    s.decodePending=new Map();
    s.decoder=new VideoDecoder({
      output:frame=>{const pending=s.decodePending.get(frame.timestamp);s.decodePending.delete(frame.timestamp);if(pending)pending.resolve(frame);else frame.close()},
      error:err=>{for(const pending of s.decodePending.values())pending.reject(err);s.decodePending.clear()}
    });
    s.decoder.configure({codec:video.avc_codec,optimizeForLatency:true});
  }
  return Promise.all(video.frames.map(f=>new Promise((resolve,reject)=>{
    const timestamp=Math.round(f.pts*1000000/24);
    s.decodePending.set(timestamp,{resolve,reject});
    s.decoder.decode(new EncodedVideoChunk({type:f.keyframe?'key':'delta',timestamp,data:f.encodedBytes}));
  })));
}
