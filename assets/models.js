// Dimensionless educational models. Plain script, not a module: browsers block
// ES modules over file:// URLs, which silently kills every demonstration when
// the page is opened from disk rather than served.
var Models=(function(){
const TAU=2*Math.PI;
function normal(rng=Math.random){return Math.sqrt(-2*Math.log(Math.max(1e-12,rng())))*Math.cos(TAU*rng())}
function wrap(x){return ((x+Math.PI)%TAU+TAU)%TAU-Math.PI}
function circularError(a,b){return Math.abs(((a-b+540)%360)-180)}
function driftStep(x,mu,sigma,dt,z){return x+mu*dt+sigma*Math.sqrt(dt)*z}
function bumpStep(theta,a,m,sigma,dt,z){return wrap(theta-a*Math.sin(m*theta)*dt+sigma*Math.sqrt(dt)*z)}
function alignmentOrder(agents){return Math.hypot(agents.reduce((a,b)=>a+Math.cos(b.theta),0),agents.reduce((a,b)=>a+Math.sin(b.theta),0))/agents.length}
function patchProbability(n,depletion){return .95*Math.exp(-depletion*n)}
function periodicDelta(d,size){return d-size*Math.round(d/size)}
function mean(a){return a.length?a.reduce((s,v)=>s+v,0)/a.length:0}
function stdev(a){if(a.length<2)return 0;const m=mean(a);return Math.sqrt(a.reduce((s,v)=>s+(v-m)**2,0)/(a.length-1))}
// Social foraging: private evidence about the patch, plus diffusive coupling to co-located foragers.
function socialBeliefStep(x,patchMean,drift,coupling,sigma,dt,z){return x+(drift+coupling*(patchMean-x))*dt+sigma*Math.sqrt(dt)*z}
function glazeUpdate(L,h){const r=Math.log((1-h)/h);return L+Math.log(Math.exp(r)+Math.exp(-L))-Math.log(Math.exp(r)+Math.exp(L))}
return {glazeUpdate,TAU,normal,wrap,circularError,driftStep,bumpStep,alignmentOrder,patchProbability,periodicDelta,mean,stdev,socialBeliefStep};
})();
