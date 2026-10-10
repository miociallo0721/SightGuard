"use strict";
const $ = (id) => document.getElementById(id);
const fileInput = $("video-file"), player = $("result-video");
let selectedFile = null, result = null, currentJob = null, currentSuggestion = null;
let processing = false, renderedFrame = -1;
let liveAudioAvailable=false;
const names = {person:"行人",bicycle:"自行车",car:"汽车",motorcycle:"摩托车",airplane:"飞机",bus:"公交车",train:"列车",truck:"卡车",boat:"船", "traffic light":"交通信号灯","fire hydrant":"消防栓","stop sign":"停车标志","parking meter":"停车计时器",bench:"长椅",bird:"鸟",cat:"猫",dog:"狗",horse:"马",sheep:"羊",cow:"牛",elephant:"大象",bear:"熊",zebra:"斑马",giraffe:"长颈鹿",backpack:"背包",umbrella:"雨伞",handbag:"手提包",tie:"领带",suitcase:"行李箱",frisbee:"飞盘",skis:"滑雪板",snowboard:"单板滑雪板","sports ball":"球",kite:"风筝","baseball bat":"棒球棒","baseball glove":"棒球手套",skateboard:"滑板",surfboard:"冲浪板","tennis racket":"网球拍",bottle:"瓶子","wine glass":"酒杯",cup:"杯子",fork:"叉子",knife:"刀",spoon:"勺子",bowl:"碗",banana:"香蕉",apple:"苹果",sandwich:"三明治",orange:"橙子",broccoli:"西兰花",carrot:"胡萝卜","hot dog":"热狗",pizza:"披萨",donut:"甜甜圈",cake:"蛋糕",chair:"椅子",couch:"沙发","potted plant":"盆栽",bed:"床","dining table":"餐桌",toilet:"马桶",tv:"显示屏",laptop:"笔记本电脑",mouse:"鼠标",remote:"遥控器",keyboard:"键盘","cell phone":"手机",microwave:"微波炉",oven:"烤箱",toaster:"烤面包机",sink:"水槽",refrigerator:"冰箱",book:"书",clock:"时钟",vase:"花瓶",scissors:"剪刀","teddy bear":"玩具熊","hair drier":"吹风机",toothbrush:"牙刷"};
function icon(name) {
  const svg = document.createElementNS("http://www.w3.org/2000/svg","svg");
  const use = document.createElementNS(svg.namespaceURI,"use");
  use.setAttribute("href",`#i-${name}`); svg.append(use); return svg;
}
function text(tag, value, className) {
  const node=document.createElement(tag); node.textContent=value;
  if(className) node.className=className; return node;
}
function plainStatus(value){return typeof value==="string"?value.replace(/\*\*(.*?)\*\*/g,"$1"):"";}
function showError(message) { $("error-message").textContent=message; $("error-message").hidden=false; }
function controls(busy) {
  processing=busy; $("analyze-button").disabled=busy||!selectedFile;
  $("example-button").disabled=busy; fileInput.disabled=busy; $("clear-file").disabled=busy;
  $("dropzone").setAttribute("aria-disabled",String(busy));
}
function setFile(file) {
  if(processing) return;
  $("error-message").hidden=true;
  if(file && !file.name.toLowerCase().endsWith(".mp4")) { showError("请选择 MP4 格式视频。"); return; }
  if(file && file.size>500*1024*1024) { showError("视频不能超过 500 MB。"); return; }
  if(file && file.size===0) { showError("所选视频文件为空。"); return; }
  selectedFile=file; $("file-summary").hidden=!file;
  if(file){$("file-name").textContent=file.name;$("file-size").textContent=`${(file.size/1024/1024).toFixed(2)} MB · 已就绪`;}
  controls(false);
}
fileInput.addEventListener("change",()=>setFile(fileInput.files[0]||null));
$("clear-file").addEventListener("click",()=>{setFile(null);fileInput.value="";});
$("dropzone").addEventListener("keydown",e=>{if((e.key==="Enter"||e.key===" ")&&!processing){e.preventDefault();fileInput.click();}});
for(const event of ["dragenter","dragover"]){$("dropzone").addEventListener(event,e=>{e.preventDefault();if(!processing)$("dropzone").classList.add("dragging");});}
for(const event of ["dragleave","drop"]){$("dropzone").addEventListener(event,e=>{e.preventDefault();$("dropzone").classList.remove("dragging");});}
$("dropzone").addEventListener("drop",e=>setFile(e.dataTransfer.files[0]||null));
async function api(url, options={}) {
  const response=await fetch(url,{cache:"no-store",...options});
  if(!response.ok){let message=`服务请求失败（${response.status}）`;try{const value=await response.json();if(typeof value.detail==="string")message=value.detail;}catch{}throw new Error(message);}
  return response.json();
}
async function postApi(url, body) {
  const options={method:"POST"};
  if(body!==undefined){options.headers={"Content-Type":"application/json"};options.body=JSON.stringify(body);}
  return api(url,options);
}
let liveActive=false, liveMuted=false, livePollTimer=null, livePollBusy=false;
let spatialActive=false, spatialPollTimer=null, spatialPollBusy=false;
let intelligenceMutationBusy=false, lastIntelligenceSnapshot=null;
function setLiveActive(active){
  liveActive=active;$("live-start").disabled=active||!$("live-camera").value;
  $("live-stop").disabled=!active;$("live-mute").disabled=!active;
  $("live-badge").textContent=active?"实时运行":"未启动";
  if(livePollTimer){clearInterval(livePollTimer);livePollTimer=null;}
  if(active)livePollTimer=setInterval(pollLive,500);
}
function setSpatialActive(active){
  spatialActive=active;
  if(spatialPollTimer){clearInterval(spatialPollTimer);spatialPollTimer=null;}
  if(active)spatialPollTimer=setInterval(pollSpatial,500);
}
function renderLiveRecords(container, rows, emptyMessage, columns=[]){
  container.replaceChildren();
  if(!Array.isArray(rows)||rows.length===0){container.append(text("p",emptyMessage));return;}
  const first=rows[0], keys=Array.isArray(first)?columns:Object.keys(first||{});
  if(!keys.length){container.append(text("p",emptyMessage));return;}
  const table=document.createElement("table"), head=document.createElement("thead"), header=document.createElement("tr");
  for(const key of keys)header.append(text("th",key));head.append(header);table.append(head);
  const body=document.createElement("tbody");
  for(const row of rows){const tr=document.createElement("tr");for(let index=0;index<keys.length;index++){
    const value=Array.isArray(row)?row[index]:row?.[keys[index]];
    const cell=document.createElement("td");cell.textContent=value==null?"—":typeof value==="object"?JSON.stringify(value):String(value);tr.append(cell);
  }body.append(tr);}table.append(body);container.append(table);
}
const liveEventColumns=["event_id","类别","track_id","状态","等级","风险评分","当前观察"];
const liveAlertColumns=["alert_id","event_id","时间 ms","类型","触发原因"];
const intelligenceModeNames={rule:"Rule Engine",laya:"Laya Structured",vision:"Laya Vision",hybrid:"Hybrid Intelligence"};
function intelligenceText(value){
  if(value===null||value===undefined||value==="")return "—";
  if(typeof value==="object")return JSON.stringify(value);
  return String(value);
}
function intelligenceMs(value){
  return typeof value==="number"&&Number.isFinite(value)?`${value.toFixed(1)} ms`:intelligenceText(value);
}
function renderIntelligence(snapshot, events){
  const data=snapshot&&typeof snapshot==="object"?snapshot:null;
  lastIntelligenceSnapshot=data;
  const mode=data?.mode in intelligenceModeNames?data.mode:"rule";
  const auxiliary=(data?.auxiliary==="vision"||mode==="vision")?"vision":"laya";
  if(!intelligenceMutationBusy){
    $("intelligence-mode").value=mode;
    $("intelligence-auxiliary").value=auxiliary;
  }
  $("intelligence-auxiliary-field").hidden=$("intelligence-mode").value!=="hybrid";
  const model=data?.model&&typeof data.model==="object"?data.model:{};
  const result=data?.latest_result&&typeof data.latest_result==="object"?data.latest_result:null;
  const comparison=data?.comparison&&typeof data.comparison==="object"?data.comparison:{};
  const performance=data?.performance&&typeof data.performance==="object"?data.performance:{};
  const activeRuleEvent=Array.isArray(events)?events.find(event=>Array.isArray(event)?event[6]==="是"&&event[4]!=="UNKNOWN":event&&event["当前观察"]==="是"&&event["等级"]!=="UNKNOWN"):null;
  const status=data?.model_status||"Unavailable";
  $("intelligence-status").textContent=status;
  $("intelligence-status").dataset.status=status;
  const selectedName=intelligenceModeNames[mode]||"辅助模型";
  const stateNotes={
    rule:"当前仅运行 Rule Engine；实际提醒由 Rule Engine 负责。",
    Unavailable:`已选择 ${selectedName} 模式，但辅助模型尚未加载；点击“加载辅助模型”后才会运行。实际提醒仍由 Rule Engine 负责。`,
    "Not Installed":"本地辅助模型或权重尚未就绪；实际提醒仍由 Rule Engine 负责。",
    Loading:"辅助模型正在加载；实际提醒仍由 Rule Engine 负责。",
    Ready:"辅助模型已加载，等待有效推理；实际提醒仍由 Rule Engine 负责。",
    Running:"辅助模型正在运行；实际提醒仍由 Rule Engine 负责。",
    Failed:"辅助模型当前不可用；实际提醒仍由 Rule Engine 负责。"
  };
  const stateKey=mode==="rule"?"rule":status;
  $("intelligence-state-note").textContent=stateNotes[stateKey]||`辅助模型状态：${status}。实际提醒仍由 Rule Engine 负责。`;
  $("intelligence-model-id").textContent=intelligenceText(model.model_id??model.id);
  $("intelligence-revision").textContent=intelligenceText(model.revision);
  $("intelligence-device").textContent=intelligenceText(model.device);
  const precision=[model.precision,model.format].filter(value=>value!==null&&value!==undefined&&value!=="").map(intelligenceText);
  $("intelligence-precision").textContent=precision.length?precision.join(" · "):"—";
  const latency=result?.inference_latency_ms??data?.inference_latency_ms??model.inference_latency_ms;
  $("intelligence-latency").textContent=intelligenceMs(latency);
  if(result&&result.valid!==false&&result.stale!==true){
    const candidate=result.candidate_risk_level;
    const score=result.candidate_score;
    const summary=candidate===null||candidate===undefined
      ?"原始判断见下方"
      :intelligenceText(candidate);
    $("intelligence-latest").textContent=score===null||score===undefined
      ?summary:`${summary} · score ${intelligenceText(score)}`;
  }else{
    $("intelligence-latest").textContent=result?(result.stale?"暂无有效结果 · 最近结果已过期":"暂无有效结果 · 最近结果不可用"):"暂无有效结果";
  }
  const auxiliaryName=intelligenceModeNames[auxiliary]||"辅助模型";
  $("intelligence-auxiliary-name").textContent=auxiliaryName;
  const ruleLevel=comparison.rule_level??(Array.isArray(activeRuleEvent)?activeRuleEvent[4]:activeRuleEvent?.["等级"]);
  $("intelligence-rule-level").textContent=intelligenceText(ruleLevel);
  $("intelligence-auxiliary-level").textContent=intelligenceText(comparison.auxiliary_level);
  const statusText=comparison.status??(comparison.agreement===true?"agreement":comparison.agreement===false?"disagreement":"unavailable");
  const comparisonLabels={agreement:"判断一致",disagreement:"判断分歧",not_comparable:"不可比较",unavailable:"等待有效且可比较的判断",stale:"结果已过期"};
  const statusKey=String(statusText).toLowerCase();
  $("intelligence-agreement").textContent=comparisonLabels[statusKey]||intelligenceText(statusText);
  $("intelligence-agreement").dataset.state=statusKey;
  $("intelligence-fusion-level").textContent=intelligenceText(comparison.fusion_level);
  const frame=comparison.frame_id??result?.frame_id;
  const event=comparison.event_id??result?.event_id;
  const timestamp=comparison.timestamp_ms??result?.timestamp_ms;
  const context=[frame===null||frame===undefined?null:`Frame ${intelligenceText(frame)}`,event===null||event===undefined?null:`Event ${intelligenceText(event)}`,timestamp===null||timestamp===undefined?null:`${intelligenceText(timestamp)} ms`].filter(Boolean);
  $("intelligence-comparison-context").textContent=context.length?context.join(" · "):"Frame / Event：—";
  const rawOutput=result?.raw_output;
  const reason=comparison.not_comparable_reason??comparison.reason??result?.error??data?.error;
  const rawParts=[];
  if(rawOutput!==null&&rawOutput!==undefined)rawParts.push(`模型原始输出：\n${typeof rawOutput==="string"?rawOutput:JSON.stringify(rawOutput,null,2)}`);
  if(reason)rawParts.push(`不可比较 / 错误原因：\n${intelligenceText(reason)}`);
  $("intelligence-raw-output").textContent=rawParts.length?rawParts.join("\n\n"):"暂无模型输出";
  const p50=performance.p50_ms,p95=performance.p95_ms;
  $("intelligence-percentiles").textContent=p50==null&&p95==null?"—":`${intelligenceMs(p50)} / ${intelligenceMs(p95)}`;
  const calls=performance.calls,timeouts=performance.timeouts;
  $("intelligence-calls").textContent=calls==null&&timeouts==null?"—":`${intelligenceText(calls)} / ${intelligenceText(timeouts)}`;
  $("intelligence-stale").textContent=intelligenceText(performance.stale);
  $("intelligence-load-time").textContent=intelligenceMs(performance.load_ms);
  $("intelligence-vram").textContent=typeof performance.peak_vram_mb==="number"&&Number.isFinite(performance.peak_vram_mb)?`${performance.peak_vram_mb.toFixed(1)} MB`:intelligenceText(performance.peak_vram_mb);
  const fpsBefore=performance.fast_fps_before,fpsAfter=performance.fast_fps_after;
  $("intelligence-fast-fps-before").textContent=typeof fpsBefore==="number"&&Number.isFinite(fpsBefore)?`${fpsBefore.toFixed(1)} FPS`:intelligenceText(fpsBefore);
  $("intelligence-fast-fps-after").textContent=typeof fpsAfter==="number"&&Number.isFinite(fpsAfter)?`${fpsAfter.toFixed(1)} FPS`:intelligenceText(fpsAfter);
  const active=liveActive&&!intelligenceMutationBusy;
  $("intelligence-mode").disabled=!active;
  $("intelligence-auxiliary").disabled=!active;
  $("intelligence-load").disabled=!active||$("intelligence-mode").value==="rule"||["Loading","Ready","Running"].includes(status);
  $("intelligence-unload").disabled=!active||!["Loading","Ready","Running","Failed"].includes(status);
}
function renderLive(view){
  if(typeof view.image==="string")$("live-image").innerHTML=view.image;
  if(typeof view.status==="string")$("live-status").textContent=plainStatus(view.status);
  if(typeof view.metrics==="string")$("live-metrics").innerHTML=view.metrics;
  if(typeof view.alert==="string")$("live-alert").innerHTML=view.alert;
  $("live-alert").classList.toggle("live-alert-rendered",typeof view.alert==="string");
  const audio=view.audio_status||null;
  const backend=audio?.audio_status||null;
  const hasWorker=Boolean(audio&&("ready" in audio||"audio_status" in audio||"running" in audio||"queued" in audio||"records" in audio));
  liveAudioAvailable=Boolean(hasWorker&&!audio.error);
  liveMuted=Boolean(hasWorker&&audio.enabled===false);
  const audioState=!hasWorker?"默认关闭":audio.error?"后端异常":audio.ready?(liveMuted?"已静音":"已启用"):"正在初始化";
  const hrtfState=backend
    ?backend.hrtf_enabled?`已启用${backend.hrtf_dataset?` · ${backend.hrtf_dataset}`:""}`:`${backend.hrtf_supported===false?"设备不支持":"未启用"}${backend.hrtf_status?` · ${backend.hrtf_status}`:""}`
    :hasWorker?"等待实际设备状态":"未初始化";
  $("live-audio-state").textContent=audioState;
  $("live-audio-backend").textContent=backend?.backend|| (hasWorker?"正在初始化":"未启动");
  $("live-audio-device").textContent=backend?.device|| (hasWorker?"正在打开设备":"未初始化");
  $("live-audio-hrtf").textContent=hrtfState;
  $("live-audio-muted").textContent=hasWorker?(liveMuted?"已静音":"未静音"):"未启动";
  const audioError=audio?.error||(backend?.fallback_reason?`后端回退：${backend.fallback_reason}`:"");
  $("live-audio-error").textContent=audioError;
  $("live-audio-error").hidden=!audioError;
  $("live-audio-debug").textContent=audio?JSON.stringify(audio,null,2):"实时空间音频默认关闭";
  renderLiveRecords($("live-events"),view.events,"尚无实时事件",liveEventColumns);
  renderLiveRecords($("live-alerts"),view.alerts,"尚无实时提醒",liveAlertColumns);
  setLiveActive(Boolean(view.running));
  renderIntelligence(view.intelligence,view.events);
  $("live-mute").textContent=liveMuted?"解除静音":"静音";
  $("live-mute").disabled=!view.running||!liveAudioAvailable||Boolean(audio?.error);
  const navigationReady=Boolean(view.running&&audio?.ready&&audio.enabled&&!audio.error);
  for(const id of ["nav-submit","nav-pause","nav-resume","nav-cancel"])$(id).disabled=!navigationReady;
  if(!navigationReady){
    $("nav-status").textContent=view.running
      ?"请在开始实时体验时开启空间音频，并等待本机设备初始化完成。"
      :"请先启动实时体验并开启空间音频；手动试听会释放实时会话。";
    $("nav-status").dataset.ready="false";
  }else if($("nav-status").dataset.ready!=="true"){
    $("nav-status").textContent="实时空间音频设备已就绪；导航操作只提交一次手动模拟提示。";
    $("nav-status").dataset.ready="true";
  }
}
async function loadLiveCameras(){
  const select=$("live-camera");
  try{
    const result=await api("/api/live/cameras");select.replaceChildren();
    for(const choice of result.choices||[]){
      if(!Array.isArray(choice)||choice.length<2)continue;
      const option=document.createElement("option");option.value=String(choice[1]);option.textContent=String(choice[0]);select.append(option);
    }
    if(result.preferred!==null&&result.preferred!==undefined)select.value=String(result.preferred);
    if(select.options.length===0)throw new Error("没有可用摄像头编号");
    $("live-start").disabled=false;
    $("live-service-status").replaceChildren(document.createElement("i"),text("span","本地服务已连接"));
  }catch(error){
    select.replaceChildren();const option=document.createElement("option");option.value="";option.textContent="无法读取摄像头列表";select.append(option);
    $("live-service-status").replaceChildren(text("span","摄像头列表读取失败"));
    $("live-start").disabled=true;$("live-error").textContent=error.message;$("live-error").hidden=false;
  }
}
async function startLive(){
  const camera=Number.parseInt($("live-camera").value,10);
  if(!Number.isInteger(camera)){showLiveError("请选择有效的摄像头编号。");return;}
  $("live-start").disabled=true;$("live-error").hidden=true;$("live-status").textContent="正在打开本机摄像头…";
  try{
    const view=await postApi("/api/live/start",{camera,spatial_audio:$("live-spatial-enabled").checked,audio_gain:Number($("live-gain").value)});
    liveMuted=false;setSpatialActive(false);renderLive(view);
    $("spatial-status").textContent="实时会话已启动；独立试听设备已释放。";
  }catch(error){showLiveError(error.message);setLiveActive(false);}
}
function showLiveError(message){$("live-error").textContent=message;$("live-error").hidden=false;}
async function pollLive(){
  if(!liveActive||livePollBusy||document.hidden)return;livePollBusy=true;
  try{renderLive(await api("/api/live/poll"));}
  catch(error){showLiveError(error.message);$("live-status").textContent="实时状态读取失败；会话将在服务端租约超时后释放。";setLiveActive(false);}
  finally{livePollBusy=false;}
}
function selectedIntelligenceAuxiliary(mode){
  if(mode==="vision")return "vision";
  if(mode==="hybrid")return $("intelligence-auxiliary").value;
  return "laya";
}
async function updateIntelligence(action){
  if(!liveActive){$("intelligence-error").textContent="请先启动当前实时会话，再管理辅助模型。";$("intelligence-error").hidden=false;return;}
  const mode=$("intelligence-mode").value;
  if(action==="load"&&mode==="rule")return;
  $("intelligence-error").hidden=true;
  intelligenceMutationBusy=true;
  renderIntelligence(lastIntelligenceSnapshot);
  try{
    const view=await postApi("/api/live/intelligence",{
      mode,auxiliary:selectedIntelligenceAuxiliary(mode),action
    });
    intelligenceMutationBusy=false;
    renderLive(view);
  }catch(error){
    intelligenceMutationBusy=false;
    renderIntelligence(lastIntelligenceSnapshot);
    $("intelligence-error").textContent=error.message;
    $("intelligence-error").hidden=false;
  }
}
$("intelligence-mode").addEventListener("change",()=>{
  $("intelligence-auxiliary-field").hidden=$("intelligence-mode").value!=="hybrid";
  void updateIntelligence("select");
});
$("intelligence-auxiliary").addEventListener("change",()=>void updateIntelligence("select"));
$("intelligence-load").addEventListener("click",()=>void updateIntelligence("load"));
$("intelligence-unload").addEventListener("click",()=>void updateIntelligence("unload"));
$("live-start").addEventListener("click",startLive);
$("live-stop").addEventListener("click",async()=>{
  $("live-stop").disabled=true;
  try{renderLive(await postApi("/api/live/stop"));liveMuted=false;setLiveActive(false);}
  catch(error){showLiveError(error.message);setLiveActive(true);}
});
$("live-mute").addEventListener("click",async()=>{
  try{renderLive(await postApi("/api/live/mute"));}
  catch(error){showLiveError(error.message);}
});
$("live-gain").addEventListener("input",()=>$("live-gain-value").textContent=Number($("live-gain").value).toFixed(2));
async function pollSpatial(){
  if(!spatialActive||spatialPollBusy||document.hidden)return;spatialPollBusy=true;
  try{renderSpatial(await api("/api/spatial/poll"));}
  catch(error){showSpatialError(error.message);setSpatialActive(false);}
  finally{spatialPollBusy=false;}
}
function renderSpatial(view){
  $("spatial-status").textContent=view.status||"";
  $("spatial-backend-status").textContent=view.backend_status?JSON.stringify(view.backend_status,null,2):"后端尚未返回设备状态";
  if(view.status&&/失败|异常停止|自动释放|已释放/.test(view.status))setSpatialActive(false);
}
function showSpatialError(message){$("spatial-error").textContent=message;$("spatial-error").hidden=false;}
async function auditionSpatial(azimuth){
  $("spatial-error").hidden=true;
  try{
    const view=await postApi("/api/spatial/play",{
      azimuth,elevation:0,gain:Number($("spatial-gain").value),cue_type:"system_test",
      device:$("spatial-device").value.trim()||null,request_hrtf:$("spatial-hrtf").checked
    });
    setSpatialActive(true);setLiveActive(false);liveMuted=false;
    $("live-status").textContent="已切换至手动空间试听，实时会话已由服务端停止。";
    $("live-audio-state").textContent="手动试听占用设备";
    $("live-audio-backend").textContent="请查看下方试听后端状态";
    $("live-audio-device").textContent="请查看下方试听后端状态";
    $("live-audio-hrtf").textContent="请查看下方试听后端状态";
    $("live-audio-muted").textContent="实时会话已释放";
    $("nav-status").textContent="导航模拟需要实时会话中的空间音频；请先停止试听，再启动实时体验并开启空间音频。";
    $("nav-status").dataset.ready="false";
    for(const id of ["nav-submit","nav-pause","nav-resume","nav-cancel"])$(id).disabled=true;
    renderSpatial(view);
  }catch(error){showSpatialError(error.message);$("spatial-status").textContent="手动测试音播放失败。";}
}
for(const button of document.querySelectorAll("[data-spatial-angle]")){
  button.addEventListener("click",()=>auditionSpatial(Number(button.dataset.spatialAngle)));
}
$("spatial-stop").addEventListener("click",async()=>{
  try{renderSpatial(await postApi("/api/spatial/stop"));}
  catch(error){showSpatialError(error.message);}
});
$("spatial-release").addEventListener("click",async()=>{
  try{renderSpatial(await postApi("/api/spatial/release"));setSpatialActive(false);}
  catch(error){showSpatialError(error.message);}
});
$("spatial-gain").addEventListener("input",()=>$("spatial-gain-value").textContent=Number($("spatial-gain").value).toFixed(2));
$("nav-angle").addEventListener("input",()=>$("nav-angle-value").textContent=`${$("nav-angle").value}°`);
async function submitNavigation(action){
  try{
    const view=await postApi("/api/live/navigation",{action,angle:Number($("nav-angle").value)});
    renderLive(view);
    $("nav-status").textContent={play:"模拟导航提示已提交一次；声音由服务器电脑播放。",pause:"模拟导航提示已暂停。",resume:"模拟导航提示已恢复。",cancel:"模拟导航提示已取消。"}[action];
    $("nav-status").dataset.ready="true";
  }catch(error){$("nav-status").textContent=`导航模拟操作失败：${error.message}`;}
}
$("nav-submit").addEventListener("click",()=>submitNavigation("play"));
$("nav-pause").addEventListener("click",()=>submitNavigation("pause"));
$("nav-resume").addEventListener("click",()=>submitNavigation("resume"));
$("nav-cancel").addEventListener("click",()=>submitNavigation("cancel"));
async function releaseHiddenSessions(){
  setLiveActive(false);setSpatialActive(false);liveMuted=false;
  try{renderLive(await postApi("/api/live/stop"));}catch(error){showLiveError(error.message);}
  try{renderSpatial(await postApi("/api/spatial/stop"));}catch(error){showSpatialError(error.message);}
  try{renderSpatial(await postApi("/api/spatial/release"));}catch(error){showSpatialError(error.message);}
}
document.addEventListener("visibilitychange",()=>{if(document.hidden)void releaseHiddenSessions();});
function sendUnloadPost(url){
  try{navigator.sendBeacon(url);}catch{}
}
window.addEventListener("pagehide",()=>{
  if("speechSynthesis" in window)speechSynthesis.cancel();
  sendUnloadPost("/api/live/stop");
  sendUnloadPost("/api/spatial/stop");
  sendUnloadPost("/api/spatial/release");
});
loadLiveCameras();
function resetReplay(){
  player.pause();player.removeAttribute("src");player.load();player.hidden=true;
  $("empty-video").hidden=false;$("video-overlay").hidden=true;
  result=null;renderedFrame=-1;currentSuggestion=null;$("speak-button").disabled=true;
  $("download-result").hidden=true;$("decision-label").textContent="正在分析";
  $("decision-reason").textContent="请等待视频处理完成。";$("decision-icon").textContent="◌";
  $("decision-card").dataset.action="unknown";$("object-list").replaceChildren(text("div","等待本次视频的识别结果","empty-results"));
  $("speech-text").textContent="等待生成提示";$("vibration-pattern").replaceChildren();
  $("vibration-text").textContent="硬件未连接 · 仅展示节奏";
  $("stat-objects").replaceChildren(text("span","—"),text("em","个 / 当前帧"));
  $("stat-frames").replaceChildren(text("span","—"),text("em","帧"));
  $("source-caption").textContent="播放时，右侧结果与视频当前帧同步更新。";
  if("speechSynthesis" in window) speechSynthesis.cancel();
}
async function run(example=false){
  if(processing||(!example&&!selectedFile))return;
  controls(true);$("error-message").hidden=true;$("progress-wrap").hidden=false;
  $("progress-bar").style.width="2%";$("progress-message").textContent=example?"正在读取已有示例…":"正在上传视频…";
  $("viewer-badge").textContent="处理中";resetReplay();
  try{
    let options={method:"POST"};
    if(!example){const data=new FormData();data.append("video",selectedFile);options.body=data;}
    const job=await api(example?"/api/example":"/api/analyze",options);currentJob=job.id;
    await poll(job.id,example);
  }catch(error){showError(error.message);$("progress-message").textContent="本次处理未完成";$("viewer-badge").textContent="分析失败";$("decision-label").textContent="暂不判断";$("decision-reason").textContent="没有本次视频的有效结果，请重新尝试。";}
  finally{controls(false);}
}
async function poll(id,example){
  while(currentJob===id){
    const state=await api(`/api/jobs/${id}`);
    $("progress-bar").style.width=`${Math.max(2,state.progress*100)}%`;
    $("progress-message").textContent=state.message;
    if(state.status==="failed")throw new Error(state.message);
    if(state.status==="completed"){
      result=await api(`/api/jobs/${id}/result`);
      player.src=result.video_url;player.hidden=false;$("empty-video").hidden=true;$("video-overlay").hidden=false;
      $("viewer-badge").textContent=example?"已保存的模型识别示例":"分析完成";
      $("stat-frames").replaceChildren(text("span",result.frames.length.toLocaleString()),text("em","帧"));
      $("source-caption").textContent=example?"已有示例仅演示识别流程，来源与道路效果未独立验证。":"当前展示标注视频；识别和方向建议随播放帧同步。";
      $("run-detail").textContent=`记录：${result.run_id}。来源：${result.source_kind}。标注视频按恒定帧率播放，原视频时间戳保留在下载结果中。`;
      $("download-result").href=`/api/jobs/${id}/download`;$("download-result").hidden=false;
      renderAt(0);return;
    }
    await new Promise(resolve=>setTimeout(resolve,650));
  }
}
$("analyze-button").addEventListener("click",()=>run(false));
$("example-button").addEventListener("click",()=>run(true));
function renderAt(seconds){
  if(!result||!result.frames.length)return;
  // The annotated preview is CFR, so match its displayed frame, not wall-clock inference time.
  const index=Math.min(result.frames.length-1,Math.max(0,Math.floor(seconds*result.fps+0.001)));
  if(index===renderedFrame)return;renderedFrame=index;
  const frame=result.frames[index], suggestion=frame.suggestion;currentSuggestion=suggestion;
  $("frame-count").textContent=`FRAME ${index+1} / ${result.frames.length}`;
  $("stat-objects").replaceChildren(text("span",frame.detections.length),text("em","个 / 当前帧"));
  const list=$("object-list");list.replaceChildren();
  if(!frame.detections.length){const empty=text("div","","empty-results");empty.append(icon("target"),text("p","当前帧未检出目标"),text("small","未检出不代表前方无障碍"));list.append(empty);}
  for(const detection of frame.detections){
    const row=text("div","","object-row"), symbol=text("span","","object-icon");
    symbol.append(icon(detection.class_name==="person"?"person":"target"));
    const name=text("div","","object-name");name.append(text("b",names[detection.class_name]||detection.class_name),text("small",`跟踪 ${detection.track_id??"未关联"} · ${detection.class_name}`));
    row.append(symbol,name,text("span",`${(detection.confidence*100).toFixed(1)}%`,"object-confidence"));list.append(row);
  }
  $("decision-card").dataset.action=suggestion.action;
  $("decision-icon").textContent={left:"↰",right:"↱",forward:"↑",stop:"⊘",unknown:"◌"}[suggestion.action];
  $("decision-label").textContent=suggestion.label;$("decision-reason").textContent=suggestion.reason;
  $("speech-text").textContent=suggestion.speech;$("speak-button").disabled=!("speechSynthesis" in window);
  const pulses=suggestion.vibration.filter((_,i)=>i%2===0);
  $("vibration-text").textContent=pulses.length?`${pulses.length===1&&pulses[0]>300?"长震一次":`短震 ${pulses.length} 次`} · 单侧马达示意` :"暂无震动编码 · 暂不判断";
  $("vibration-pattern").replaceChildren();
  for(const pulse of pulses){const p=text("span","");p.style.width=`${Math.max(15,pulse/9)}px`;$("vibration-pattern").append(p);}
  $("vibration-pattern").setAttribute("aria-label",$("vibration-text").textContent);
}
for(const event of ["timeupdate","seeked","loadedmetadata"]){player.addEventListener(event,()=>renderAt(player.currentTime));}
if("requestVideoFrameCallback" in player){const tick=(_now,metadata)=>{renderAt(metadata.mediaTime);player.requestVideoFrameCallback(tick);};player.requestVideoFrameCallback(tick);}
player.addEventListener("error",()=>{if(player.getAttribute("src"))showError("视频加载失败，请重新加载示例或上传浏览器支持的 MP4。");});
$("speak-button").addEventListener("click",()=>{
  if(!currentSuggestion||!("speechSynthesis" in window))return;
  speechSynthesis.cancel();const voice=new SpeechSynthesisUtterance(currentSuggestion.speech);
  voice.lang="zh-CN";voice.rate=.95;voice.onerror=()=>showError("当前浏览器语音不可用，可查看右侧文字提示。");
  speechSynthesis.speak(voice);
});
window.addEventListener("pagehide",()=>{if("speechSynthesis" in window)speechSynthesis.cancel();});
api("/api/status").then(status=>{
  const label=status.model_available?"本地模型已就绪":"请先准备本地模型";
  $("service-status").replaceChildren(document.createElement("i"),text("span",label));
}).catch(()=>{$("service-status").textContent="本地服务未连接";});
const links=Array.from(document.querySelectorAll("nav a"));
const observer=new IntersectionObserver(entries=>{
  for(const entry of entries){if(entry.isIntersecting){links.forEach(link=>link.classList.toggle("active",link.hash===`#${entry.target.id}`));}}
},{rootMargin:"-15% 0px -60% 0px"});
for(const id of ["home","demo","live-experience","product","features","scenarios","team"])observer.observe($(id));
