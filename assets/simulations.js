(function(){
const {normal,wrap,driftStep,bumpStep,socialBeliefStep,glazeUpdate,mean,stdev,TAU}=Models;
const $=id=>document.getElementById(id),val=id=>Number($(id).value),reduced=matchMedia('(prefers-reduced-motion: reduce)').matches;
const colors=['#d8bd82','#8fb8a6','#c08a3e','#c9d2ae'];
const sims={};
function setup(id,reset,step,draw){const canvas=$(id+'-canvas'),ctx=canvas.getContext('2d');const s={id,canvas,ctx,playing:!reduced,reset,step,draw,t:0};sims[id]=s;reset(s);draw(s);const toggle=document.querySelector(`[data-toggle="${id}"]`);toggle.textContent=s.playing?'Pause':'Play';toggle.addEventListener('click',()=>{s.playing=!s.playing;toggle.textContent=s.playing?'Pause':'Play'});document.querySelector(`[data-reset="${id}"]`).addEventListener('click',()=>{reset(s);draw(s)});return s}
function bg(s){s.ctx.fillStyle='#1f3a31';s.ctx.fillRect(0,0,800,430);s.ctx.strokeStyle='#37544a';s.ctx.lineWidth=1;for(let x=60;x<=760;x+=100){s.ctx.beginPath();s.ctx.moveTo(x,30);s.ctx.lineTo(x,380);s.ctx.stroke()}for(let y=60;y<400;y+=60){s.ctx.beginPath();s.ctx.moveTo(50,y);s.ctx.lineTo(760,y);s.ctx.stroke()}s.ctx.font='12px Arial';s.ctx.fillStyle='#c3cfc4'}
setup('decisions',s=>{s.t=0;s.state=1;s.perfect=0;s.norm=0;s.trace=[];s.pOK=0;s.nOK=0;s.n=0;s.wait=0},(s,dt)=>{s.t+=dt;const h=val('world-change')*dt,ha=Math.max(1e-6,val('assumed-change')*dt);if(val('world-change')>0&&Math.random()<h)s.state=-s.state;const llr=val('drift')*s.state*dt+val('noise')*Math.sqrt(dt)*normal();s.perfect+=llr;s.norm=glazeUpdate(s.norm,ha)+llr;s.n++;if(Math.sign(s.perfect)===s.state)s.pOK++;if(Math.sign(s.norm)===s.state)s.nOK++;s.trace.push([s.t,s.perfect,s.norm,s.state]);if(s.trace.length>520)s.trace.shift();},s=>{bg(s);const c=s.ctx,W=680,X=60;if(!s.trace.length)return;const t0=s.trace[0][0],t1=Math.max(t0+1,s.trace[s.trace.length-1][0]);const x=t=>X+(t-t0)/(t1-t0)*W,CLIP=6,y=v=>215-Math.max(-CLIP,Math.min(CLIP,v))/CLIP*150;let i0=0;for(let i=1;i<=s.trace.length;i++){if(i===s.trace.length||s.trace[i][3]!==s.trace[i0][3]){const up=s.trace[i0][3]>0,xa=x(s.trace[i0][0]),xb=x(s.trace[i-1][0]);const w=Math.max(1,xb-xa);c.fillStyle=up?'rgba(216,189,130,.15)':'rgba(143,184,166,.15)';c.fillRect(xa,44,w,351);c.fillStyle=up?'#d8bd82':'#8fb8a6';c.fillRect(xa,36,w,6);if(w>26){c.font='17px Arial';c.fillText(up?'\u2191':'\u2193',(xa+xb)/2-5,64);c.font='12px Arial'}i0=i}}c.strokeStyle='#8fb8a6';c.setLineDash([4,6]);c.beginPath();c.moveTo(X,y(0));c.lineTo(X+W,y(0));c.stroke();c.setLineDash([]);[[1,'#d8bd82'],[2,'#c9922f']].forEach(([k,col])=>{c.strokeStyle=col;c.lineWidth=2;c.beginPath();s.trace.forEach((p,i)=>i?c.lineTo(x(p[0]),y(p[k])):c.moveTo(x(p[0]),y(p[k])));c.stroke()});c.fillStyle='#c3cfc4';c.fillText('shaded bands show which source is active',X,28);c.fillStyle='#c3cfc4';c.fillText('favours source A',X,y(0)-8);c.fillText('favours source B',X,y(0)+18);const p=s.n?Math.round(100*s.pOK/s.n):0,nn=s.n?Math.round(100*s.nOK/s.n):0;c.font='10px Arial';c.fillStyle='#d8bd82';c.fillText('NO DISCOUNTING (CLASSICAL)',X,404);c.fillStyle='#c9922f';c.fillText('DISCOUNTING THE PAST',X+340,404);c.font='24px Georgia';c.fillStyle='#d8bd82';c.fillText(p+'% accurate',X,426);c.fillStyle='#c9922f';c.fillText(nn+'% accurate',X+340,426);c.font='12px Arial';$('decisions-metric').textContent=`Accuracy: ${p}% without discounting, ${nn}% with discounting`});
// Changing evidence or boundaries restarts trajectories so one run has one model.
['drift','noise','world-change','assumed-change'].forEach(id=>$(id).addEventListener('input',()=>{sims.decisions.reset(sims.decisions);sims.decisions.draw(sims.decisions)}));
const PATCHES=5,FORAGERS=25,LEAVE=-1,START=1.1;
function patchX(i){return 90+i*(620/(PATCHES-1))}
setup('collective',s=>{s.t=0;s.patches=Array.from({length:PATCHES},()=>({resource:.55+Math.random()*.45}));s.agents=Array.from({length:FORAGERS},(_,i)=>({patch:i%PATCHES,belief:START,travel:0,from:0,trace:[]}));s.lastLeave=new Array(PATCHES).fill(-99);s.gaps=[]},(s,dt)=>{s.t+=dt;const g=val('coupling'),sigma=val('forage-noise'),travelTime=val('patch-travel');
 const here=s.patches.map((_,p)=>s.agents.filter(a=>a.travel<=0&&a.patch===p));
 // Patches deplete while occupied and recover once abandoned.
 s.patches.forEach((patch,p)=>{patch.resource=Math.max(0,Math.min(1,patch.resource-.055*here[p].length*dt+(here[p].length?0:.12*dt)))});
 const means=here.map(group=>mean(group.map(a=>a.belief)));
 for(const a of s.agents){
  if(a.travel>0){a.travel-=dt;a.trace.push(null);if(a.travel<=0){a.belief=START;a.trace.length=0}continue}
  const drift=2*s.patches[a.patch].resource-1.15;
  a.belief=socialBeliefStep(a.belief,means[a.patch],drift,g,sigma,dt,normal());
  a.trace.push(a.belief);
  if(a.belief<=LEAVE){
   const others=[...Array(PATCHES).keys()].filter(p=>p!==a.patch);
   const gap=s.t-s.lastLeave[a.patch];if(gap<8)s.gaps.push(gap);s.lastLeave[a.patch]=s.t;
   if(s.gaps.length>120)s.gaps.shift();
   a.from=a.patch;a.patch=others[Math.floor(Math.random()*others.length)];a.travel=travelTime;
  }
 }

 for(const a of s.agents)if(a.trace.length>240)a.trace.shift();
},s=>{bg(s);const c=s.ctx;const here=s.patches.map((_,p)=>s.agents.filter(a=>a.travel<=0&&a.patch===p));
 c.fillStyle='#c3cfc4';c.fillText('PATCHES',60,25);
 s.patches.forEach((patch,p)=>{const x=patchX(p),top=70,h=90;
  c.strokeStyle='#567064';c.lineWidth=1.5;c.beginPath();c.roundRect(x-52,top,104,h,8);c.stroke();
  c.fillStyle='#8fb8a628';c.beginPath();c.roundRect(x-52,top+h*(1-patch.resource),104,h*patch.resource,8);c.fill();
  here[p].forEach((a,i)=>{c.fillStyle=colors[a.patch%4];c.beginPath();c.arc(x-38+(i%7)*13,top+16+Math.floor(i/7)*20,4.2,0,TAU);c.fill()});
  c.fillStyle='#c3cfc4';c.fillText(`${here[p].length}`,x-4,top+h+16)});
 // foragers in transit
 s.agents.filter(a=>a.travel>0).forEach(a=>{const u=1-a.travel/Math.max(.01,val('patch-travel'));const x=patchX(a.from)+(patchX(a.patch)-patchX(a.from))*u;c.fillStyle='#d8bd82';c.globalAlpha=.55;c.beginPath();c.arc(x,196,3.6,0,TAU);c.fill();c.globalAlpha=1});
 c.fillStyle='#c3cfc4';c.fillText('IN TRANSIT',60,200);
 c.fillText('BELIEF THAT THE PATCH IS STILL WORTH IT',60,240);
 const yb=v=>330-v*52;
 c.strokeStyle='#d8bd82';c.setLineDash([4,6]);c.beginPath();c.moveTo(60,yb(LEAVE));c.lineTo(740,yb(LEAVE));c.stroke();c.setLineDash([]);
 c.fillText('leave',60,yb(LEAVE)+16);
 s.agents.forEach(a=>{c.strokeStyle=colors[a.patch%4];c.globalAlpha=.8;c.lineWidth=1.6;c.beginPath();let pen=false;a.trace.forEach((v,i)=>{if(v===null){pen=false;return}const x=60+i/239*680,y=yb(v);pen?c.lineTo(x,y):c.moveTo(x,y);pen=true});c.stroke()});
 c.globalAlpha=1;c.fillStyle='#c3cfc4';
 const groups=here.filter(gr=>gr.length>1);
 if(groups.length){const inst=mean(groups.map(gr=>stdev(gr.map(a=>a.belief))));s.spread=s.spread===undefined?inst:.96*s.spread+.04*inst}
 const pct=s.gaps.length>8?Math.round(100*s.gaps.filter(x=>x<.5).length/s.gaps.length):0;
 $('collective-metric').textContent=`${FORAGERS} foragers · ${PATCHES} patches · within-patch belief spread ${(s.spread??0).toFixed(2)} · ${pct}% of departures follow another within 0.5 s`});
// Social escape: group of accumulators with social discounting alpha (dimensionless caricature).
const E={th:1.6,mu:1,sig:.8,lam:3,kick:.35,T:6,AX0:60,AX1:540,AY0:44,AY1:206};
function escapeMC(N,alpha,trials=400,dt=.02){
 const run=(n,a,H)=>{const x=new Float32Array(n);for(let k=0;k*dt<E.T;k++){const f=1,leak=a*E.lam*f*dt,drift=(H?E.mu:0)*dt,sd=E.sig*Math.sqrt(dt);for(let i=0;i<n;i++){x[i]+=drift-leak*x[i]+sd*normal();if(x[i]>=E.th)return k*dt}}return null};
 const out=(n,a)=>{let fa=0,det=0,hit=0;for(let k=0;k<trials;k++){if(run(n,a,false)!==null)fa++;const d=run(n,a,true);if(d!==null){det+=d;hit++}}return{fa:fa/trials,det:hit?det/hit:E.T}};
 return{lone:out(1,0),group:out(N,alpha)}}
let escMC=null,escTimer=0;
function refreshEscapeMC(){clearTimeout(escTimer);escTimer=setTimeout(()=>{escMC=escapeMC(val('escape-n'),val('escape-alpha'));const g=escMC.group,l=escMC.lone;$('escape-metric').textContent=`${val('escape-n')} animals · α = ${val('escape-alpha').toFixed(2)} · false alarms ${Math.round(100*g.fa)}% (alone ${Math.round(100*l.fa)}%) · detection ${g.det.toFixed(2)} s (alone ${l.det.toFixed(2)} s)`},120)}
function escapeLayout(s){const n=val('escape-n');s.fish=Array.from({length:n},(_,i)=>{const r=Math.sqrt((i+.5)/n),a=i*2.39996;const bx=(E.AX0+E.AX1)/2+r*190*Math.cos(a),by=(E.AY0+E.AY1)/2+r*52*Math.sin(a);return{bx,by,x:bx,y:by,vx:0,vy:0,h:Math.random()*TAU,b:0,fled:false,fx:0,fy:0}})}
function escapeTrial(s){s.H=Math.random()<.5;s.clock=0;s.first=null;s.pause=0;s.outcome='';s.trace=[];s.fish.forEach(f=>{f.b=0;f.fled=false});s.nf=0}
setup('escape',s=>{escapeLayout(s);escapeTrial(s);s.t=0;refreshEscapeMC()},(s,dt)=>{
 const n=s.fish.length,a=val('escape-alpha');s.t+=dt;
 for(const f of s.fish){ // calm swimming around the school's home position; fled animals scatter
  if(f.fled){f.x+=f.fx*dt;f.y+=f.fy*dt;f.h=Math.atan2(f.fy,f.fx);continue}
  f.vx+=(f.bx-f.x)*.9*dt+normal()*14*dt;f.vy+=(f.by-f.y)*.9*dt+normal()*14*dt;f.vx*=.96;f.vy*=.96;f.x+=f.vx*dt*4;f.y+=f.vy*dt*4;f.h=Math.atan2(f.vy,f.vx)||f.h}
 if(s.pause>0){s.pause-=dt;if(s.pause<=0)escapeTrial(s);return}
 s.clock+=dt;const present=s.H&&s.clock>.4;let newF=0;const frac=1-s.nf/n;
 s.fish.forEach((f,i)=>{if(f.fled)return;f.b+=((present?E.mu:0)-a*E.lam*frac*f.b)*dt+E.sig*Math.sqrt(dt)*normal();if(f.b>=E.th){f.fled=true;newF++;const ang=Math.atan2(f.y-(E.AY0+E.AY1)/2,f.x-(E.AX0+E.AX1)/2)+normal()*.5;f.fx=Math.cos(ang)*(90+Math.random()*50);f.fy=Math.sin(ang)*(50+Math.random()*30);f.at=s.clock}});
 if(newF){if(s.first===null)s.first=s.clock;s.nf+=newF;s.fish.forEach(f=>{if(!f.fled)f.b+=E.kick*newF})}
 if(s.trace.length<8)for(let i=s.trace.length;i<Math.min(8,n);i++)s.trace.push([]);
 s.trace.forEach((tr,i)=>{if(s.fish[i]&&(!s.fish[i].fled||s.fish[i].at>=s.clock-.02))tr.push([s.clock,s.fish[i].b])});
 if(s.first!==null&&s.clock>s.first+1.4){s.outcome=s.H?`DETECTED IN ${s.first.toFixed(2)} s`:'FALSE ALARM';s.pause=1.1}
 else if(s.clock>E.T){s.outcome=s.H?'MISSED':'CORRECTLY STILL';s.pause=.9}
},s=>{
 const c=s.ctx,n=s.fish.length;bg(s);
 // arena
 const g=c.createLinearGradient(0,E.AY0,0,E.AY1);g.addColorStop(0,'#27493d');g.addColorStop(1,'#1b342b');c.fillStyle=g;c.beginPath();c.roundRect(E.AX0,E.AY0,E.AX1-E.AX0,E.AY1-E.AY0,10);c.fill();
 c.save();c.beginPath();c.roundRect(E.AX0,E.AY0,E.AX1-E.AX0,E.AY1-E.AY0,10);c.clip();
 // threat: a bird's shadow crossing the arena
 if(s.H&&s.clock>.2){const u=Math.min(1,(s.clock-.2)/4),bx=E.AX0-50+u*(E.AX1-E.AX0+100),by=E.AY0+40,gr=c.createRadialGradient(bx,by,2,bx,by,52);gr.addColorStop(0,'rgba(8,18,14,.55)');gr.addColorStop(1,'rgba(8,18,14,0)');c.fillStyle=gr;c.fillRect(bx-60,by-60,120,120)}
 // animals
 for(const f of s.fish){const col=f.fled?'#d8bd82':'#8fb8a6';c.save();c.translate(f.x,f.y);c.rotate(f.h);c.fillStyle=col;c.globalAlpha=f.fled?.95:.55+Math.min(.4,f.b>0?f.b*.3:0);c.beginPath();c.moveTo(11,0);c.lineTo(-8,-5.4);c.lineTo(-5,0);c.lineTo(-8,5.4);c.closePath();c.fill();c.restore()}
 c.restore();c.strokeStyle='#567064';c.lineWidth=1.2;c.beginPath();c.roundRect(E.AX0,E.AY0,E.AX1-E.AX0,E.AY1-E.AY0,10);c.stroke();
 const tag=(L,t,x,y)=>{c.font='bold 13px Arial';c.fillStyle='#e8eee6';c.fillText(L,x,y);c.font='11px Arial';c.fillStyle='#c3cfc4';c.fillText(t,x+16,y)};tag('A','GROUP',E.AX0,E.AY0-10);
 c.textAlign='right';c.font='11px Arial';c.fillStyle=s.H?'#d8bd82':'#8fb8a6';c.fillText(s.H?'THREAT PRESENT':'NO THREAT',E.AX1,E.AY0-10);
 if(s.outcome){c.font='11px Arial';c.fillStyle=/FALSE|MISSED/.test(s.outcome)?'#e3a07a':'#d8bd82';c.fillText(s.outcome,E.AX1-12,E.AY1-12)}
 c.textAlign='left';c.font='12px Arial';
 // belief traces
 tag('B','BELIEF x OF EIGHT ANIMALS THAT A THREAT IS PRESENT',60,248);
 const X0=60,W=680,yb=v=>380-Math.max(-.6,Math.min(2,v))*50,xt=t=>X0+t/4*W;
 c.strokeStyle='#37544a';c.lineWidth=1;c.beginPath();c.moveTo(X0,yb(0));c.lineTo(X0+W,yb(0));c.stroke();
 c.strokeStyle='#d8bd82';c.setLineDash([4,6]);c.beginPath();c.moveTo(X0,yb(E.th));c.lineTo(X0+W,yb(E.th));c.stroke();c.setLineDash([]);c.fillStyle='#d8bd82';c.font='italic 12px Georgia';c.fillText('θ',X0-14,yb(E.th)+4);c.fillStyle='#c3cfc4';c.fillText('0',X0-13,yb(0)+4);c.font='11px Arial';for(let t=0;t<=4;t++){c.strokeStyle='#567064';c.beginPath();c.moveTo(xt(t),yb(0));c.lineTo(xt(t),yb(0)+5);c.stroke();c.fillText(String(t),xt(t)-3,yb(0)+18)}c.fillText('time (s)',X0+W-44,yb(0)+34);
 s.trace.forEach((tr,i)=>{if(tr.length<2)return;const fl=s.fish[i]&&s.fish[i].fled;c.strokeStyle=fl?'#d8bd82':'#8fb8a6';c.globalAlpha=.85;c.lineWidth=1.6;c.beginPath();tr.forEach(([t,v],k)=>k?c.lineTo(xt(t),yb(v)):c.moveTo(xt(t),yb(v)));c.stroke();if(fl){const l=tr[tr.length-1];c.fillStyle='#d8bd82';c.beginPath();c.arc(xt(l[0]),yb(l[1]),3.4,0,TAU);c.fill()}});c.globalAlpha=1;
 // comparison with a lone animal
 if(escMC){const bx=566,bw=118;c.font='bold 13px Arial';c.fillStyle='#e8eee6';c.fillText('C',bx,E.AY0-10);c.font='11px Arial';c.fillStyle='#c3cfc4';c.fillText('GROUP VS. LONE ANIMAL',bx+16,E.AY0-10);c.font='12px Arial';[['#6f8f82','LONE ANIMAL',0],['#d8bd82','GROUP',1]].forEach(([col,name,k])=>{c.fillStyle=col;c.beginPath();c.roundRect(bx,E.AY0+4+k*18,10,10,2);c.fill();c.fillStyle='#c3cfc4';c.fillText(name,bx+16,E.AY0+13+k*18)});
  const row=(label,y,lone,grp,max,fmt)=>{c.font='11px Arial';c.fillStyle='#c3cfc4';c.fillText(label,bx,y);[[lone,'#6f8f82'],[grp,'#d8bd82']].forEach(([v,col],k)=>{const yy=y+8+k*21;c.fillStyle='#2d4a40';c.beginPath();c.roundRect(bx,yy,bw,15,4);c.fill();c.fillStyle=col;c.beginPath();c.roundRect(bx,yy,Math.max(4,bw*Math.min(1,v/max)),15,4);c.fill();c.font='14px Georgia';c.fillStyle='#e8eee6';c.fillText(fmt(v),bx+bw+8,yy+13)})};
  row('FALSE ALARMS',E.AY0+66,escMC.lone.fa,escMC.group.fa,1,v=>Math.round(100*v)+'%');row('DETECTION TIME',E.AY0+130,escMC.lone.det,escMC.group.det,3,v=>v.toFixed(2)+' s')}
});
$('escape-n').addEventListener('input',()=>{sims.escape.reset(sims.escape);sims.escape.draw(sims.escape)});
$('escape-alpha').addEventListener('input',refreshEscapeMC);
setup('memory',s=>{s.theta=.7;s.history=[];s.t=0},(s,dt)=>{s.t+=dt;s.theta=bumpStep(s.theta,val('pinning'),val('wells'),val('diffusion'),dt,normal());s.history.push(s.theta);if(s.history.length>450)s.history.shift()},s=>{bg(s);const c=s.ctx;const curve=[];for(let i=0;i<=300;i++){const theta=-Math.PI+i/300*TAU;curve.push([60+i/300*680,190-125*Math.exp(-.5*(wrap(theta-s.theta)/.34)**2)])}c.beginPath();c.moveTo(60,190);curve.forEach(([x,y])=>c.lineTo(x,y));c.lineTo(740,190);c.closePath();c.fillStyle='#8fb8a635';c.fill();c.beginPath();curve.forEach(([x,y],i)=>i?c.lineTo(x,y):c.moveTo(x,y));c.strokeStyle='#a8c6b1';c.lineWidth=3;c.stroke();c.strokeStyle='#d8bd82';c.setLineDash([3,5]);if(val('pinning')>0)for(let j=0;j<val('wells');j++){let theta=wrap(TAU*j/val('wells'));let x=60+(theta+Math.PI)/TAU*680;c.beginPath();c.moveTo(x,35);c.lineTo(x,195);c.stroke()}c.setLineDash([]);c.fillText('NEURAL ACTIVITY',60,25);c.fillText('−180°',60,214);c.fillText('180°',708,214);c.fillText('RECENT BUMP POSITION',60,260);c.strokeStyle='#c9922f';c.lineWidth=2;c.beginPath();s.history.forEach((theta,i)=>{const x=60+i/449*680,y=330-theta/Math.PI*52;if(i===0||Math.abs(theta-s.history[i-1])>Math.PI)c.moveTo(x,y);else c.lineTo(x,y)});c.stroke();c.fillText('older',60,410);c.fillText('now →',692,410);$('memory-metric').textContent=`Bump center ${(s.theta*180/Math.PI).toFixed(1)}° · ${val('wells')} preferred locations · ${s.t.toFixed(1)} model seconds`});
let last=0,acc=0,frameCount=0;
function safely(s,fn){try{fn()}catch(err){if(!s.broken){s.broken=true;console.error(`simulation "${s.id}" stopped:`,err)}}}
function frame(now){const dt=Math.min((now-last)/1000,.05);last=now;if(!document.hidden){acc+=dt;while(acc>=1/60){for(const s of Object.values(sims))if(s.playing&&!s.broken)safely(s,()=>s.step(s,1/60));acc-=1/60}if(++frameCount%2===0)for(const s of Object.values(sims))if(!s.broken)safely(s,()=>s.draw(s))}requestAnimationFrame(frame)}requestAnimationFrame(frame);
document.documentElement.dataset.simsReady='1';
})();
