"""Single-page web client: chat, spoken input/output via the server's local models."""

PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Omarchy Agent</title>
<style>
:root{color-scheme:dark light;--bg:#111;--fg:#e8e8e8;--mut:#888;--acc:#7aa2f7;--box:#1b1b1b}
@media(prefers-color-scheme:light){:root{--bg:#fafafa;--fg:#1a1a1a;--box:#eee}}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 ui-monospace,monospace;
display:flex;flex-direction:column;height:100dvh}
#log{flex:1;overflow:auto;padding:12px 16px}
.m{margin:8px 0;white-space:pre-wrap}.m pre{background:var(--box);padding:8px;overflow:auto}.m code{background:var(--box);padding:0 3px}.you{color:var(--acc)}.err{color:#f7768e}.sys{color:var(--mut)}
#bar{display:flex;gap:8px;padding:12px 16px;border-top:1px solid var(--box)}
#in{flex:1;background:var(--box);color:var(--fg);border:0;padding:10px;font:inherit}
button{background:var(--box);color:var(--fg);border:0;padding:10px 14px;font:inherit;cursor:pointer}
button.rec{background:#f7768e;color:#000}
#appr{display:none;padding:12px 16px;background:#3b2a00;color:#ffd479}
#appr button{margin-left:8px}
</style></head><body>
<div id="appr"><span id="adesc"></span><button id="yes">Approve</button><button id="no">Deny</button></div>
<div id="log"></div>
<div id="bar"><input id="in" placeholder="message" autocomplete="off">
<button id="mic">Mic</button><button id="send">Send</button>
<label class="sys"><input type="checkbox" id="spk"> speak</label></div>
<script>
const token=new URLSearchParams(location.hash.slice(1)).get("token")||"";
history.replaceState(null,"",location.pathname);
const H={Authorization:"Bearer "+token};
const log=document.getElementById("log"),inp=document.getElementById("in");
function esc(t){return t.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;")}
function md(t){t=esc(t);
t=t.replace(/```[a-z]*\n?([\s\S]*?)```/g,(m,c)=>"<pre>"+c+"</pre>");
t=t.replace(/`([^`\n]+)`/g,"<code>$1</code>");
t=t.replace(/\*\*([^*\n]+)\*\*/g,"<b>$1</b>");
t=t.replace(/(^|[\s(])\*([^*\n]+)\*/g,"$1<i>$2</i>");
t=t.replace(/^#{1,6}\s+(.*)$/gm,"<b>$1</b>");
t=t.replace(/^[-*]\s+/gm,"• ");return t}
function setMd(el,t){el.innerHTML=md(t)}
function add(cls,t){const d=document.createElement("div");d.className="m "+cls;
if(cls===""){setMd(d,t)}else{d.textContent=t}log.appendChild(d);log.scrollTop=log.scrollHeight;return d}
if(!token)add("err","No token. Open this page as /#token=YOUR_TOKEN");
let aid=null;
async function poll(){try{const r=await fetch("/api/approvals",{headers:H});const j=await r.json();
const a=(j.pending||[])[0];const box=document.getElementById("appr");
if(a){aid=a.id;document.getElementById("adesc").textContent="Approve? "+a.description;box.style.display="block"}else{aid=null;box.style.display="none"}}catch(e){}}
setInterval(poll,1500);
async function decide(ok){if(aid==null)return;await fetch("/api/approvals/"+aid,{method:"POST",headers:H,body:JSON.stringify({approve:ok})});poll()}
document.getElementById("yes").onclick=()=>decide(true);
document.getElementById("no").onclick=()=>decide(false);
async function send(){const text=inp.value.trim();if(!text)return;inp.value="";add("you","you: "+text);
const out=add("","");let full="";
try{const r=await fetch("/api/chat",{method:"POST",headers:H,body:JSON.stringify({prompt:text,stream:true})});
const rd=r.body.getReader(),dec=new TextDecoder();let buf="";
for(;;){const {done,value}=await rd.read();if(done)break;buf+=dec.decode(value,{stream:true});
let i;while((i=buf.indexOf("\n\n"))>=0){const line=buf.slice(0,i).replace(/^data: /,"");buf=buf.slice(i+2);
if(line==="[DONE]")continue;let j;try{j=JSON.parse(line)}catch{continue}
if(j.delta){full+=j.delta;setMd(out,full)}if(j.error){add("err","error: "+j.error)}if(j.done){full=j.answer||full;setMd(out,full)}}}
}catch(e){add("err",String(e))}
if(full&&document.getElementById("spk").checked)speak(full)}
async function speak(t){const r=await fetch("/api/speak",{method:"POST",headers:H,body:JSON.stringify({text:t})});
if(r.ok){new Audio(URL.createObjectURL(await r.blob())).play()}}
let rec=null,chunks=[];
document.getElementById("mic").onclick=async function(){
if(rec){rec.stop();return}
try{const s=await navigator.mediaDevices.getUserMedia({audio:true});rec=new MediaRecorder(s);chunks=[];
rec.ondataavailable=e=>chunks.push(e.data);this.classList.add("rec");
rec.onstop=async()=>{s.getTracks().forEach(t=>t.stop());this.classList.remove("rec");rec=null;
const r=await fetch("/api/transcribe",{method:"POST",headers:H,body:new Blob(chunks)});
const j=await r.json();if(j.text){inp.value=j.text;send()}else add("err",j.error||"no speech")};
rec.start()}catch(e){add("err","mic: "+e)}};
document.getElementById("send").onclick=send;
inp.addEventListener("keydown",e=>{if(e.key==="Enter")send()});
</script></body></html>
"""
