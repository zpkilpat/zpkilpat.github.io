(function(){
const {normal,wrap,driftStep,bumpStep,socialBeliefStep,mean,stdev,TAU}=Models;
const $=id=>document.getElementById(id),val=id=>Number($(id).value),reduced=matchMedia('(prefers-reduced-motion: reduce)').matches;
const colors=['#d8bd82','#8fb8a6','#c08a3e','#c9d2ae'];
const sims={};
function setup(id,reset,step,draw){const canvas=$(id+'-canvas'),ctx=canvas.getContext('2d');const s={id,canvas,ctx,playing:!reduced,reset,step,draw,t:0};sims[id]=s;reset(s);draw(s);const toggle=document.querySelector(`[data-toggle="${id}"]`);toggle.textContent=s.playing?'Pause':'Play';toggle.addEventListener('click',()=>{s.playing=!s.playing;toggle.textContent=s.playing?'Pause':'Play'});document.querySelector(`[data-reset="${id}"]`).addEventListener('click',()=>{reset(s);draw(s)});return s}
function bg(s){s.ctx.fillStyle='#1f3a31';s.ctx.fillRect(0,0,800,430);s.ctx.strokeStyle='#37544a';s.ctx.lineWidth=1;for(let x=60;x<=760;x+=100){s.ctx.beginPath();s.ctx.moveTo(x,30);s.ctx.lineTo(x,380);s.ctx.stroke()}for(let y=60;y<400;y+=60){s.ctx.beginPath();s.ctx.moveTo(50,y);s.ctx.lineTo(760,y);s.ctx.stroke()}s.ctx.font='12px Arial';s.ctx.fillStyle='#c3cfc4'}
setup('decisions',s=>{s.t=0;s.paths=Array.from({length:22},()=>({x:0,points:[[0,0]],done:false,choice:0,time:0}));s.wait=0},(s,dt)=>{s.t+=dt;const threshold=val('threshold');for(const p of s.paths){if(p.done)continue;p.x=driftStep(p.x,val('drift'),val('noise'),dt,normal());if(Math.abs(p.x)>=threshold){p.x=Math.sign(p.x)*threshold;p.done=true;p.choice=Math.sign(p.x);p.time=s.t}p.points.push([s.t,p.x])}if(s.t>15){s.wait+=dt;if(s.wait>1)s.reset(s)}},s=>{bg(s);const c=s.ctx,threshold=val('threshold'),scale=150/Math.max(2,threshold),x=t=>60+t/15*680,y=v=>215-v*scale;c.setLineDash([6,6]);c.strokeStyle='#d8bd82';for(const b of [-threshold,threshold]){c.beginPath();c.moveTo(60,y(b));c.lineTo(740,y(b));c.stroke()}c.setLineDash([]);c.fillText('CHOICE A',64,y(threshold)-10);c.fillText('CHOICE B',64,y(-threshold)+20);s.paths.forEach((p,i)=>{c.strokeStyle=colors[i%4];c.globalAlpha=.75;c.lineWidth=1.8;c.beginPath();p.points.forEach(([t,v],j)=>j?c.lineTo(x(t),y(v)):c.moveTo(x(t),y(v)));c.stroke()});c.globalAlpha=1;c.fillStyle='#c3cfc4';c.fillText('0',58,410);c.fillText('15 · model time',655,410);const done=s.paths.filter(p=>p.done);$('decisions-metric').textContent=`${done.length} / 22 decisions · ${done.filter(p=>p.choice>0).length} choice A · ${done.filter(p=>p.choice<0).length} choice B${done.length?' · mean time '+(done.reduce((a,p)=>a+p.time,0)/done.length).toFixed(2):''}`});
// Changing evidence or boundaries restarts trajectories so one run has one model.
['drift','noise','threshold'].forEach(id=>$(id).addEventListener('input',()=>{sims.decisions.reset(sims.decisions);sims.decisions.draw(sims.decisions)}));
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
setup('memory',s=>{s.theta=.7;s.history=[];s.t=0},(s,dt)=>{s.t+=dt;s.theta=bumpStep(s.theta,val('pinning'),val('wells'),val('diffusion'),dt,normal());s.history.push(s.theta);if(s.history.length>450)s.history.shift()},s=>{bg(s);const c=s.ctx;const curve=[];for(let i=0;i<=300;i++){const theta=-Math.PI+i/300*TAU;curve.push([60+i/300*680,190-125*Math.exp(-.5*(wrap(theta-s.theta)/.34)**2)])}c.beginPath();c.moveTo(60,190);curve.forEach(([x,y])=>c.lineTo(x,y));c.lineTo(740,190);c.closePath();c.fillStyle='#8fb8a635';c.fill();c.beginPath();curve.forEach(([x,y],i)=>i?c.lineTo(x,y):c.moveTo(x,y));c.strokeStyle='#a8c6b1';c.lineWidth=3;c.stroke();c.strokeStyle='#d8bd82';c.setLineDash([3,5]);if(val('pinning')>0)for(let j=0;j<val('wells');j++){let theta=wrap(TAU*j/val('wells'));let x=60+(theta+Math.PI)/TAU*680;c.beginPath();c.moveTo(x,35);c.lineTo(x,195);c.stroke()}c.setLineDash([]);c.fillText('NEURAL ACTIVITY',60,25);c.fillText('−180°',60,214);c.fillText('180°',708,214);c.fillText('RECENT BUMP POSITION',60,260);c.strokeStyle='#c9922f';c.lineWidth=2;c.beginPath();s.history.forEach((theta,i)=>{const x=60+i/449*680,y=330-theta/Math.PI*52;if(i===0||Math.abs(theta-s.history[i-1])>Math.PI)c.moveTo(x,y);else c.lineTo(x,y)});c.stroke();c.fillText('older',60,410);c.fillText('now →',692,410);$('memory-metric').textContent=`Bump center ${(s.theta*180/Math.PI).toFixed(1)}° · ${val('wells')} preferred locations · ${s.t.toFixed(1)} model seconds`});
let last=0,acc=0,frameCount=0;
function safely(s,fn){try{fn()}catch(err){if(!s.broken){s.broken=true;console.error(`simulation "${s.id}" stopped:`,err)}}}
function frame(now){const dt=Math.min((now-last)/1000,.05);last=now;if(!document.hidden){acc+=dt;while(acc>=1/60){for(const s of Object.values(sims))if(s.playing&&!s.broken)safely(s,()=>s.step(s,1/60));acc-=1/60}if(++frameCount%2===0)for(const s of Object.values(sims))if(!s.broken)safely(s,()=>s.draw(s))}requestAnimationFrame(frame)}requestAnimationFrame(frame);
document.documentElement.dataset.simsReady='1';
})();
