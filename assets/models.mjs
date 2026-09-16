// Dimensionless educational models. Pure functions are shared by demos and tests.
export const TAU=2*Math.PI;
export function normal(rng=Math.random){return Math.sqrt(-2*Math.log(Math.max(1e-12,rng())))*Math.cos(TAU*rng())}
export function wrap(x){return ((x+Math.PI)%TAU+TAU)%TAU-Math.PI}
export function circularError(a,b){return Math.abs(((a-b+540)%360)-180)}
export function driftStep(x,mu,sigma,dt,z){return x+mu*dt+sigma*Math.sqrt(dt)*z}
export function bumpStep(theta,a,m,sigma,dt,z){return wrap(theta-a*Math.sin(m*theta)*dt+sigma*Math.sqrt(dt)*z)}
export function alignmentOrder(agents){return Math.hypot(agents.reduce((a,b)=>a+Math.cos(b.theta),0),agents.reduce((a,b)=>a+Math.sin(b.theta),0))/agents.length}
export function patchProbability(n,depletion){return .95*Math.exp(-depletion*n)}
export function periodicDelta(d,size){return d-size*Math.round(d/size)}
export function mean(a){return a.length?a.reduce((s,v)=>s+v,0)/a.length:0}
export function stdev(a){if(a.length<2)return 0;const m=mean(a);return Math.sqrt(a.reduce((s,v)=>s+(v-m)**2,0)/(a.length-1))}
// Social foraging: private evidence about the patch, plus diffusive coupling to co-located foragers.
export function socialBeliefStep(x,patchMean,drift,coupling,sigma,dt,z){return x+(drift+coupling*(patchMean-x))*dt+sigma*Math.sqrt(dt)*z}
