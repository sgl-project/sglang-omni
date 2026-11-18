'use strict';
const el=id=>document.getElementById(id);
const delay=ms=>new Promise(ok=>setTimeout(ok,ms));
const period=17/24;
const playbackLeadSeconds=.015;
let current=null,available=false,mediaResetAvailable=false,mediaResetLookahead=0;
let presets={},portraitData='';
const fields=['appearance','environment','initial_pose'];
let promptPrefix='',promptSuffix='';
function fullSystemPrompt(){return promptPrefix+fields.map(f=>el(f).value.trim()).join(' ')+promptSuffix}
function updateSystemPrompt(){el('systemPrompt').value=promptPrefix?fullSystemPrompt():''}
function selectPreset(){const k=el('preset').value;fields.forEach(f=>el(f).value=presets[k][f]);portraitData='';el('photo').value='';el('portrait').src='./portraits/'+k+'.jpg';updateSystemPrompt()}
async function loadPresets(){presets=await(await fetch('./presets')).json();for(const [k,p] of Object.entries(presets))el('preset').add(new Option(p.label,k));selectPreset();controls()}
el('preset').onchange=selectPreset;
el('photo').onchange=async()=>{const file=el('photo').files[0];if(!file){portraitData='';el('portrait').src='./portraits/'+el('preset').value+'.jpg';return}
try{const bmp=await createImageBitmap(file),c=document.createElement('canvas');c.width=576;c.height=768;const scale=Math.max(576/bmp.width,768/bmp.height);c.getContext('2d').drawImage(bmp,(576-bmp.width*scale)/2,(768-bmp.height*scale)/2,bmp.width*scale,bmp.height*scale);bmp.close();el('portrait').src=c.toDataURL('image/jpeg',.94);portraitData=el('portrait').src.split(',')[1]}catch{el('status').textContent='无法读取参考图，请重新选择'}};
fields.forEach(f=>el(f).oninput=updateSystemPrompt);
el('randomSeed').onclick=()=>{el('seed').value=crypto.getRandomValues(new Uint32Array(1))[0]};
el('import').onchange=async()=>{try{const data=JSON.parse(await el('import').files[0].text());if(!fields.every(f=>typeof data[f]==='string'&&data[f].trim()&&data[f].length<=800))throw Error();fields.forEach(f=>el(f).value=data[f]);updateSystemPrompt()}catch{el('status').textContent='设定 JSON 无效，请提供人物、环境和初始姿态字段'}};
function personaPacket(){const seed=el('seed').valueAsNumber;if(!Number.isInteger(seed)||seed<0||seed>4294967295)throw Error('随机种子必须是 0 到 4294967295 之间的整数');const p={preset:el('preset').value,image:portraitData,prompt_mode:'fields',seed};for(const f of fields){p[f]=el(f).value.trim();if(!p[f])throw Error('请填写人物、环境和初始姿态')}return p}
function controls(){const busy=!!current;
el('editor').disabled=busy;
el('live').disabled=el('sample').disabled=busy||!available||!Object.keys(presets).length;
el('stop').disabled=!busy;
el('resetMedia').disabled=!mediaResetAvailable||!current?.received||!!current.resetInProgress;}
async function health(){if(current)return;try{const s=await(await fetch('./status',{cache:'no-store'})).json();if(current)return;mediaResetAvailable=!!s.media_reset_supported;mediaResetLookahead=s.media_reset_lookahead_blocks||0;promptPrefix=s.system_prompt_prefix||'';promptSuffix=s.system_prompt_suffix||'';updateSystemPrompt();available=s.ready&&!s.busy;el('status').textContent=s.busy?'已有会话进行中':s.ready?'服务就绪':'模型准备中…'}catch{available=false;if(!current)el('status').textContent='服务连接中…'}controls()}
function send(s,obj){if(current===s&&s.ws?.readyState===1)s.ws.send(JSON.stringify(obj))}
function end(s,message){if(current!==s)return;send(s,{type:'session.close'});current=null;clearInterval(s.firstFrameTimer);if(s.decoder&&s.decoder.state!=='closed')s.decoder.close();for(const frame of s.frames)frame.img.close?.();s.ws?.close();s.stream?.getTracks().forEach(t=>t.stop());s.audio?.close().catch(()=>{});s.inputAudio?.close().catch(()=>{});const v=el('input');v.classList.remove('live-mirror');v.pause();v.srcObject=null;v.removeAttribute('src');v.load();el('inHint').hidden=false;el('status').textContent=message;controls()}
function encode(arr){const bytes=new Uint8Array(arr.buffer,arr.byteOffset,arr.byteLength);let data='';for(let i=0;i<bytes.length;i+=8192)data+=String.fromCharCode(...bytes.subarray(i,i+8192));return btoa(data)}
async function cameraBlock(s,n){const capturedAt=performance.now(),frames=[],target=Math.ceil((n+1)*17*16000/24),start=Math.ceil(n*17*16000/24);let uploaded=false,uploadedFrameCount=0;
  for(let i=0;i<17;i++){await delay(Math.max(0,s.tick-performance.now()));if(current!==s)return;s.tick+=1000/24;
    const v=el('input'),w=v.videoWidth,h=v.videoHeight,ratio=320/384,cw=Math.min(w,h*ratio),ch=cw/ratio;
    s.ctx.drawImage(v,(w-cw)/2,(h-ch)/2,cw,ch,0,0,320,384);frames.push(s.capture.toDataURL('image/jpeg',.76).split(',')[1]);
    if((i%2===1||i===16)&&(uploaded||(!s.queue.length&&!s.resetRequested))){
      send(s,{type:'input_audio_buffer.append',audio:'',format:'pcm_f32le',sample_rate_hz:16000,is_speech:false,video_frames:frames.slice(uploadedFrameCount)});
      uploaded=true;uploadedFrameCount=frames.length;
    }}
  while(current===s && s.mic.length<target-start)await delay(4);
  if(current!==s)return;
  const pcm=Float32Array.from(s.mic.splice(0,target-start));
  const rms=Math.sqrt(pcm.reduce((a,x)=>a+x*x,0)/pcm.length),speech=rms>.007;
  return {frames,samples:pcm,speech,capturedAt,uploaded}}
async function captureLive(s){
  for(let n=0;current===s;n++){
    if(n===2){
      s.capturing=false;
      while(current===s&&!s.firstFrameShown)await delay(15);
      if(current!==s)return;
      s.mic.length=0;s.tick=performance.now();s.capturing=true;
    }
    const block=await cameraBlock(s,n);
    if(!block||current!==s)return;
    s.queue.push(block);
    while(s.queue.length>2){
      const stale=s.queue.findIndex((item,i)=>i<s.queue.length-1&&!item.speech&&!item.uploaded);
      if(stale<0)break;
      s.queue.splice(stale,1);s.droppedInputs=(s.droppedInputs||0)+1;
    }
    if(s.queue.length>=16){
      s.captureStopped=true;s.capturing=false;s.stream.getTracks().forEach(t=>t.stop());
      el('status').textContent='推理落后，采集已暂停，正在播完已录内容';return;
    }
    el('inputlabel').textContent=(block.speech?'检测到语音':'当前安静')+' · 待提交 '+s.queue.length+' 块';
  }
}
async function produce(s){let start=performance.now();let previewStarted=false;
  if(s.mode==='live'){s.queue=[];s.capturing=true;captureLive(s).catch(err=>end(s,err.message))}
  for(let n=0;current===s&&(s.mode==='live'||n<s.count);n++){
    if(n===2){
      el('inputlabel').textContent='等待首个画面，暂停输入';
      if(s.mode==='sample')el('input').pause();
      while(current===s&&!s.firstFrameShown)await delay(15);
      if(current!==s)return;
      el('inputlabel').textContent=s.mode==='live'?'摄像头与麦克风':'内置对话样片';
      if(s.mode==='sample'){await el('input').play();start=performance.now()-n*period*1000}
    }
    while(current===s&&(s.resetRequested||s.sent-s.received>=2))await delay(15);
    if(current!==s)return;
    let block;
    if(s.mode==='sample'){
      if(!previewStarted){await el('input').play();previewStarted=true}

      await delay(Math.max(0,start+(n+1)*period*1000-performance.now()));
      if(current!==s)return;
      while(current===s&&s.resetRequested)await delay(15);if(current!==s)return;
      s.captureTimes.push(performance.now()-period*1000);s.submitted.push(performance.now());s.sent++;send(s,{type:'demo.sample_block',index:n});continue;
    }else{
      while(current===s&&!(block=nextInputBlock(s.queue,el('catchup').checked,s.audio,s.captureStopped))){
        if(s.captureStopped&&!s.queue.length){s.captureCount=s.sent;return}await delay(8)
      }
      if(block?.accelerated)s.inputCatchups=(s.inputCatchups||0)+1;
    }
    if(!block||current!==s)return;
    while(current===s&&s.resetRequested)await delay(15);if(current!==s)return;
    if(!block.uploaded)for(let i=0;i<17;i+=2)send(s,{type:'input_audio_buffer.append',audio:'',format:'pcm_f32le',sample_rate_hz:16000,is_speech:false,video_frames:block.frames.slice(i,i+2)});
    const pcm16=new Int16Array(block.samples.length);
    for(let i=0;i<pcm16.length;i++)pcm16[i]=Math.fround(Math.max(-1,Math.min(1,block.samples[i]))*32767);
    const audio=encode(pcm16);
    if(block.speech)send(s,{type:'input_audio_buffer.append',audio,format:'pcm_s16le',sample_rate_hz:16000,is_speech:true,video_frames:[]});
    else send(s,{type:'demo.record_audio',audio,format:'pcm_s16le'});
    s.captureTimes.push(block.capturedAt);s.submitted.push(performance.now());s.sent++;send(s,{type:'input_audio_buffer.commit',final:true})
  }}
function unpackPacket(buffer){
  const headerLength=new DataView(buffer).getUint32(0,true);
  const packet=JSON.parse(new TextDecoder().decode(new Uint8Array(buffer,4,headerLength)));
  let offset=4+headerLength;
  packet.pcmBytes=new Uint8Array(buffer,offset,packet.audio_byte_length);offset+=packet.audio_byte_length;
  for(const frame of packet.video.frames){frame.encodedBytes=new Uint8Array(buffer,offset,frame.byte_length);offset+=frame.byte_length}
  return packet;
}
async function playback(s,p){const bytes=p.pcmBytes||Uint8Array.from(atob(p.delta),c=>c.charCodeAt(0)),pcm=new Int16Array(bytes.slice().buffer);
  const block=p.video.chunk_index,frameOffset=p.video.frame_offset||0,blockComplete=p.video.block_complete!==false;
  const ctx=s.audio;let buf=ctx.createBuffer(2,pcm.length/2,32000);
  for(let ch=0;ch<2;ch++){const data=buf.getChannelData(ch);for(let j=0;j<data.length;j++)data[j]=pcm[j*2+ch]/32768}
  const images=await decodeVideoFrames(s,p.video);
  if(current!==s)return;
  const backlog=Math.max(0,s.playEnd-ctx.currentTime);
  const rate=frameOffset?s.blockPlaybackRates[block]:(el('catchup').checked&&backlog>.10?1+Math.min(.12,(backlog-.08)*.25):1);
  if(!frameOffset)s.blockPlaybackRates[block]=rate;
  const originalLength=buf.length;buf=catchupAudio(buf,rate,ctx);
  const effectiveRate=originalLength/buf.length;
  const t=Math.max(ctx.currentTime+playbackLeadSeconds,s.playEnd);s.playEnd=t+buf.duration;if(!frameOffset)s.playWait[block]=Math.max(0,t-ctx.currentTime)*1000;
  el('buffer').textContent='播放前等待 '+Math.round(Math.max(0,t-ctx.currentTime)*1000)+' ms · 含当前块缓冲 '+Math.round(Math.max(0,s.playEnd-ctx.currentTime)*1000)+' ms'+(effectiveRate>1.001?' · 追赶 '+effectiveRate.toFixed(2)+'×':'');

  const src=ctx.createBufferSource();src.buffer=buf;src.connect(ctx.destination);src.start(t);
  images.forEach((img,i)=>s.frames.push({t:t+i/(24*effectiveRate),img,capturedAt:s.captureTimes[block]+(frameOffset+i)*1000/24,block:p.video.chunk_index}));
  src.onended=()=>{if(current!==s||!blockComplete)return;send(s,{type:'playback.ack',played_ms:Math.floor((block+1)*period*1000),committed_ms:Math.floor((block+1)*period*1000),response_id:p.response_id});if(s.mode==='sample'&&block+1===s.count)end(s,'示例播放完毕');if(s.captureCount&&block+1===s.captureCount)end(s,'已播完全部采集内容')};
}
function paint(s){if(current!==s)return;let frame;while(s.frames.length&&s.frames[0].t<=s.audio.currentTime){frame?.img.close?.();frame=s.frames.shift();}if(frame){el('output').getContext('2d').drawImage(frame.img,0,0,576,768);frame.img.close?.();el('outHint').hidden=true;s.firstFrameShown=true;
    if(s.displayedBlock!==frame.block){
      s.displayedBlock=frame.block;
      const latency=Math.round(performance.now()-frame.capturedAt),buffered=Math.round(Math.max(0,s.playEnd-s.audio.currentTime)*1000),queued=s.queue?.length||0,inflight=s.sent-s.received;
      el('latency').textContent='画面延迟 '+latency+' ms · 待提交 '+queued+' 块 · 处理中 '+inflight+' 块';
      send(s,{type:'demo.client_stats',block:frame.block,display_latency_ms:latency,playback_buffer_ms:buffered,input_queue_blocks:queued,input_inflight_blocks:inflight,server_submit_to_av_ms:s.serverTimes[frame.block]??null,submit_to_receive_ms:Math.round(s.receivedAt[frame.block]-s.submitted[frame.block]),capture_to_submit_ms:Math.round(s.submitted[frame.block]-s.captureTimes[frame.block]),receive_to_display_ms:Math.round(performance.now()-s.receivedAt[frame.block]),scheduled_play_wait_ms:Math.round(s.playWait[frame.block]||0),input_catchup_blocks:s.inputCatchups||0,dropped_silent_inputs:s.droppedInputs||0});
    }
  }requestAnimationFrame(()=>paint(s))}
async function start(mode){if(current)return;el('resetStatus').textContent='';el('streams').hidden=false;const s={mode,sent:0,received:0,submitted:[],receivedAt:[],serverTimes:[],playWait:[],blockPlaybackRates:[],captureTimes:[],playEnd:0,mic:[],frames:[],chain:Promise.resolve()};current=s;controls();el('outHint').hidden=false;el('status').textContent='正在建立会话…';el('stats').textContent='等待首个音视频块';el('output').getContext('2d').clearRect(0,0,576,768);
  try{const persona=personaPacket();s.audio=new AudioContext();await s.audio.resume();s.h264=new URLSearchParams(location.search).get('video')!=='jpeg'&&await h264Supported();const v=el('input');v.controls=false;
    if(mode==='live'){
      v.classList.add('live-mirror');
      s.stream=await navigator.mediaDevices.getUserMedia({video:{width:{ideal:576},height:{ideal:768},frameRate:{ideal:24}},audio:{echoCancellation:true,noiseSuppression:true,channelCount:1}});
      if(current!==s){s.stream.getTracks().forEach(t=>t.stop());return}
      v.muted=true;v.srcObject=s.stream;await v.play();
      s.capture=document.createElement('canvas');s.capture.width=320;s.capture.height=384;s.ctx=s.capture.getContext('2d');
      s.inputAudio=new AudioContext();await s.inputAudio.audioWorklet.addModule('./capture.js');await s.inputAudio.resume();
      const source=s.inputAudio.createMediaStreamSource(s.stream),mic=new AudioWorkletNode(s.inputAudio,'mic-capture'),mute=s.inputAudio.createGain();mute.gain.value=0;source.connect(mic).connect(mute).connect(s.inputAudio.destination);
      mic.port.onmessage=e=>{if(s.capturing)s.mic.push(...e.data)};s.inputNodes=[source,mic,mute]
    }else{v.classList.remove('live-mirror');v.srcObject=null;v.src='./example.mp4';v.muted=false;v.volume=.45;v.load()}
    el('inputlabel').textContent=mode==='live'?'摄像头与麦克风':'内置对话样片';el('inHint').hidden=true;
    s.count=(await(await fetch('./status')).json()).blocks;if(current!==s)return;
    const ws=s.ws=new WebSocket(new URL('./ws?setup=1&binary=1&stream=1'+(s.h264?'&video=h264':''),location.href).href.replace(/^http/,'ws'));
    ws.binaryType='arraybuffer';ws.onopen=()=>{send(s,persona);el('status').textContent='正在准备人物和模型…'};
    ws.onmessage=e=>{if(current!==s)return;const p=typeof e.data==='string'?JSON.parse(e.data):unpackPacket(e.data);
      if(p.type==='session.updated'&&s.resetRequested){s.resetRequested=false;el('resetStatus').textContent='已到发布边界，将在下一块开始前重置';controls()}
      if(p.type==='demo.preparing'&&!s.sessionReady)el('status').textContent=p.message+'… '+p.elapsed_s+' 秒';
      if(p.type==='session.created'){s.sessionReady=true;s.firstFrameStarted=performance.now();el('status').textContent='正在准备首个画面…';s.firstFrameTimer=setInterval(()=>{if(current===s&&!s.received)el('status').textContent='正在准备首个画面… '+Math.floor((performance.now()-s.firstFrameStarted)/1000)+' 秒'},1000);s.tick=performance.now();s.mic.length=0;paint(s);produce(s).catch(err=>end(s,err.message))}
      if(p.type==='response.output_audio.delta'&&p.video){clearInterval(s.firstFrameTimer);el('status').textContent='会话进行中';const n=p.video.chunk_index;if(!p.video.frame_offset){s.receivedAt[n]=performance.now();s.serverTimes[n]=p.video.timings?.submit_to_av_ms;}if(p.video.block_complete!==false)s.received++;
      if(p.video.timings?.realization_kv_reset_generation){s.resetInProgress=false;el('resetStatus').textContent='已在第 '+(p.video.chunk_index+1)+' 块前重置 Realization 历史 KV，保留 Intent 与 Codec 状态'}controls();el('stats').textContent='已返回 '+s.received+' 块 · '+(Number.isFinite(s.serverTimes[n])?'服务端 '+Math.round(s.serverTimes[n])+' ms · ':'')+'提交到返回 '+Math.round(performance.now()-s.submitted[n])+' ms';s.chain=s.chain.then(()=>playback(s,p)).catch(err=>end(s,'播放失败：'+err.message))}
      if(p.type==='error')end(s,p.error?.message||'服务返回错误')};
    ws.onerror=()=>end(s,'连接失败，请确认服务就绪后重试');ws.onclose=()=>end(s,'会话已关闭')
  }catch(err){end(s,'无法开始：'+err.message)}}
async function resetMediaCache(){
  const s=current;
  if(!mediaResetAvailable||!s||s.resetInProgress)return;
  s.resetRequested=s.resetInProgress=true;
  controls();
  el('resetStatus').textContent='等待已提交的视频块发布完成…';
  while(current===s&&s.sent>s.received+mediaResetLookahead)await delay(15);
  if(current===s)send(s,{type:'demo.reset_media'});
}
el('live').onclick=()=>start('live');
el('sample').onclick=()=>start('sample');
el('stop').onclick=()=>current&&end(current,'会话已结束');
el('resetMedia').onclick=resetMediaCache;
window.addEventListener('pagehide',()=>current&&end(current,'已关闭'));
controls();
loadPresets().catch(()=>el('status').textContent='预设加载失败');
health();
setInterval(health,5000);
