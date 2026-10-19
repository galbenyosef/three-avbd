/* Copyright (c) 2026 Chris Giles
 * Permission to use, copy, modify, distribute and sell this software and its
 * documentation for any purpose is hereby granted without fee, provided that
 * the above copyright notice appear in all copies. Chris Giles makes no
 * representations about the suitability of this software for any purpose.
 * It is provided "as is" without express or implied warranty.
 */
import { sdfSupportsSource } from './wgsl-sdf-supports.ts';
import { SDF_FIELDS_WGSL } from './wgsl-sdf-fields.ts';
import { SPHERE_FEATURE } from './layout.ts';
import { SDF_MAX_POINTS } from './sdf-layout.ts';
/** Standalone Algorithm 1 pass: one invocation per pair, four cached + one new
 * candidate, independent per-point normals.
 */
/** sdfNarrowphaseWGSL for a pass whose pair results hold `maxPoints` points. */
export const sdfNarrowphaseSource = (maxPoints: number = SDF_MAX_POINTS) => /* wgsl */ `
struct Params { counts: vec4u, values: vec4f, distances: vec4f, flags: vec4u }
struct Body { position: vec4f, rotation: vec4f, velocity: vec4f, omega: vec4f, lo: vec4f, hi: vec4f, anchorRotation: vec4f }
struct Point { x: vec4f, normal: vec4f, anchorA: vec4f, anchorB: vec4f, flags: vec4u }
struct Result { identity: vec4u, stats: vec4u, points: array<Point,${maxPoints}> }
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage,read> atlas: array<vec4u>;
@group(0) @binding(2) var<storage,read> bodies: array<Body>;
@group(0) @binding(3) var<storage,read> pairs: array<vec4u>;
@group(0) @binding(4) var<storage,read> previous: array<Result>;
@group(0) @binding(5) var<storage,read_write> results: array<Result>;
var<private> work: Result;
var<private> bodyA: Body;
var<private> bodyB: Body;
var<private> lo: vec3f;
var<private> hi: vec3f;
var<private> rng: u32;
${SDF_FIELDS_WGSL}
// Per-pair/step 32-bit RNG. Intentional scheduling-independent GPU policy, not
// the CPU oracle's global 64-bit PCG stream. Seed selection is tested separately.
fn hash(value: u32) -> u32 { var x=value; x=(x^(x>>16u))*0x7feb352du; x=(x^(x>>15u))*0x846ca68bu; return x^(x>>16u); }
fn randomU32() -> u32 { rng=hash(rng+0x9e3779b9u); return rng; }
fn randomFloat() -> f32 { return f32(randomU32()>>8u)/16777216.0; }
fn smoothMax(a: f32,b: f32) -> f32 { return 0.5*(a+b+sqrt((a-b)*(a-b)+params.values.y)); }
fn objective(p: vec3f) -> f32 {
  work.stats.x++; var value=smoothMax(worldField(bodyA,p).phi,worldField(bodyB,p).phi);
  for(var i=0u;i<work.identity.w;i++) { value -= params.values.z*length(p-work.points[i].x.xyz); }
  return value;
}
fn optimize(seed: vec3f, step: vec3f) -> vec3f {
  var x: array<vec3f,4>; var values: array<f32,4>; x[0]=clamp(seed,lo,hi);
  for(var i=0u;i<3u;i++) { x[i+1u]=x[0]; x[i+1u][i] += select(-step[i],step[i],x[0][i]+step[i]<=hi[i]); x[i+1u]=clamp(x[i+1u],lo,hi); }
  for(var i=0u;i<4u;i++) { values[i]=objective(x[i]); }
  var iterations=0u;
  for(;iterations<params.flags.x;iterations++) {
    for(var i=1u;i<4u;i++) { var j=i; loop { if(j==0u) { break; } if(values[j]>=values[j-1u]) { break; }
      let v=values[j]; values[j]=values[j-1u]; values[j-1u]=v; let point=x[j]; x[j]=x[j-1u]; x[j-1u]=point; j--; } }
    let extent=max(length(x[1]-x[0]),max(length(x[2]-x[0]),length(x[3]-x[0])));
    if(extent<params.values.w || values[3]-values[0]<0.05*params.values.w) { break; }
    let centroid=(x[0]+x[1]+x[2])/3.0; let reflected=clamp(centroid+(centroid-x[3]),lo,hi); let fr=objective(reflected);
    if(fr<values[0]) { let expanded=clamp(centroid+2.0*(centroid-x[3]),lo,hi); let fe=objective(expanded);
      x[3]=select(reflected,expanded,fe<fr); values[3]=min(fe,fr);
    } else if(fr<values[2]) { x[3]=reflected; values[3]=fr; }
    else { let contracted=centroid+0.5*(select(x[3],reflected,fr<values[3])-centroid); let fc=objective(contracted);
      if(fc<min(fr,values[3])) { x[3]=contracted; values[3]=fc; }
      else { for(var i=1u;i<4u;i++) { x[i]=x[0]+0.5*(x[i]-x[0]); values[i]=objective(x[i]); } }
    }
  }
  work.stats.y += iterations; var best=0u; for(var i=1u;i<4u;i++) { if(values[i]<values[best]) { best=i; } } return x[best];
}
fn inside(p: vec3f) -> bool { return all(p>=lo) && all(p<=hi); }
fn seedPoint() -> vec3f {
  // Explicit draws avoid depending on constructor-argument evaluation ordering.
  let rx=randomFloat(); let ry=randomFloat(); let rz=randomFloat(); var p=lo+(hi-lo)*vec3f(rx,ry,rz); var n=0u;
  if(params.flags.z==0u) { return p; }
  for(var b=0u;b<2u;b++) {
    var body=bodyA; if(b==1u) { body=bodyB; } let header=atlas[bodyHeader(body)];
    for(var i=0u;i<header.w;i++) {
      let candidate=worldPoint(body,bitcast<vec4f>(atlas[header.z+i]).xyz);
      if(inside(candidate)) { n++; if(randomU32()%n==0u) { p=candidate; } }
    }
  }
  return p;
}
fn velocity(body: Body,p: vec3f) -> vec3f { return body.velocity.xyz+cross(body.omega.xyz,p-body.position.xyz); }
fn addPoint(old: Point, seed: vec3f, cached: bool) {
  var step=max(0.1*(hi-lo),vec3f(0.5*params.values.w));
  if(cached) { step=min(step,vec3f(max(length(seed-old.x.xyz),5.0*params.values.w))); }
  let p=optimize(seed,step); let a=worldField(bodyA,p); let b=worldField(bodyB,p); work.stats.x++;
  let na=unit(a.gradient); let nb=-unit(b.gradient); var normal=select(nb,na,a.phi<b.phi);
  if(cached) { normal=select(nb,na,dot(na,old.normal.xyz)>dot(nb,old.normal.xyz)); }
  let offsetA=a.gradient*(a.phi/max(dot(a.gradient,a.gradient),1e-8)); let offsetB=b.gradient*(b.phi/max(dot(b.gradient,b.gradient),1e-8));
  var separation=a.phi+b.phi;
  if(a.phi>0.0 || b.phi>0.0) {
    let gap=offsetA-offsetB; separation=length(gap);
    // Unsigned distance misclassifies an embedded surface as a positive gap.
    if(params.distances.z!=0.0 && (a.phi<=0.0 || b.phi<=0.0)) {
      let fixedA=bodyA.velocity.w!=0.0 && atlas[bodyHeader(bodyA)].x==0u;
      let fixedB=bodyB.velocity.w!=0.0 && atlas[bodyHeader(bodyB)].x==0u;
      if(fixedA || fixedB) {
        normal=select(nb,na,fixedA); separation=dot(gap,normal);
        // phi/|grad|^2 is unbounded where a voxel gradient vanishes; fall back to the bounded sum.
        let moving=select(a.gradient,b.gradient,fixedA); if(dot(moving,moving)<0.25) { separation=a.phi+b.phi; }
      }
    }
    if(a.phi>0.0 && b.phi>0.0 && separation>1e-6 && !(cached && dot(gap,old.normal.xyz)<0.0)) { normal=gap/separation; }
  }
  // New ids stay below SDF_ROLLING_BIT and never equal SPHERE_FEATURE, which evalContact
  // would otherwise treat as a built-in sphere contact.
  var id=old.flags.z; if(!cached) { work.stats.z++; if(work.stats.z==${SPHERE_FEATURE}u) { work.stats.z++; }
    if(work.stats.z==0u || work.stats.z>=0x20000000u) { work.stats.z=1u; } id=work.stats.z; }
  let point=Point(vec4f(p,min(a.phi,b.phi)),vec4f(normal,separation),
    vec4f(rot(bodyA.anchorRotation,invrot(bodyA.rotation,p-offsetA-bodyA.position.xyz)),smoothMax(a.phi,b.phi)),
    vec4f(rot(bodyB.anchorRotation,invrot(bodyB.rotation,p-offsetB-bodyB.position.xyz)),0),vec4u(u32(separation<params.distances.x),u32(cached),id,0));
  work.points[work.identity.w]=point; work.identity.w++;
}
${sdfSupportsSource(maxPoints)}
@compute @workgroup_size(64)
fn narrowphase(@builtin(global_invocation_id) invocation: vec3u) {
  let slot=invocation.x; if(slot>=params.counts.x) { return; }
  let pair=pairs[slot]; work.identity=vec4u(pair.xyz,0u); work.stats=vec4u(0);
  if(pair.x==0xffffffffu) { results[slot]=work; return; }
  bodyA=bodies[pair.x]; bodyB=bodies[pair.y]; let margin=vec3f(0.5*params.distances.x);
  // Rolling surfaces must refresh anchors rather than pinning last frame's material point: analytic
  // sphere, capsule, cylinder, cone and torus (atlas kinds 1-5). A pinned anchor sank rolling cylinders.
  let kindA=atlas[bodyHeader(bodyA)].x; let kindB=atlas[bodyHeader(bodyB)].x;
  work.stats.w=u32((kindA>=1u && kindA<=5u) || (kindB>=1u && kindB<=5u));
  lo=max(bodyA.lo.xyz-margin,bodyB.lo.xyz-margin); hi=min(bodyA.hi.xyz+margin,bodyB.hi.xyz+margin);
  if(any(lo>hi) || (bodyA.velocity.w!=0.0 && bodyB.velocity.w!=0.0)) { results[slot]=work; return; }
  rng=hash(pair.z^hash(params.counts.y)^hash(pair.x)^hash(pair.y+0x9e3779b9u));
  var cached=0u; var activeCount=0u;
  if(params.flags.y!=0u && pair.w!=0xffffffffu && pair.w<arrayLength(&previous)) {
    let old=previous[pair.w];
    if(all(old.identity.xyz==pair.xyz)) {
      work.stats.z=old.stats.z;
      for(var i=0u;i<min(old.identity.w,${maxPoints - 1}u);i++) {
        let point=old.points[i];
        if(point.flags.z>=0x40000000u) { continue; }
        if(cached>=params.counts.z) { break; }
        cached++; activeCount += point.flags.x;
        let va=velocity(bodyA,point.x.xyz); let vb=velocity(bodyB,point.x.xyz); var v=0.5*(va+vb);
        if(bodyA.velocity.w!=0.0) { v=vb; } else if(bodyB.velocity.w!=0.0) { v=va; }
        addPoint(point,point.x.xyz+v*params.values.x,true);
      }
    }
  }
  let skip=params.flags.w!=0u && activeCount>0u && activeCount==params.counts.z && atlas[bodyHeader(bodyA)].y!=0u && atlas[bodyHeader(bodyB)].y!=0u;
  let samples=select(params.counts.w,0u,skip)+params.counts.z-cached;
  for(var i=0u;i<samples;i++) { var empty: Point; addPoint(empty,seedPoint(),false); }
  // First-wins duplicate rejection, then stable ranking (Algorithm 1).
  var unique=0u;
  for(var i=0u;i<work.identity.w;i++) {
    let point=work.points[i]; var duplicate=false;
    for(var j=0u;j<unique;j++) { let d=point.x.xyz-work.points[j].x.xyz; if(dot(d,d)<params.values.w*params.values.w) { duplicate=true; } }
    if(!duplicate) { work.points[unique]=point; unique++; }
  }
  work.identity.w=unique;
  for(var i=0u;i<unique;i++) {
    var rank=work.points[i].anchorA.w+select(params.distances.y,0.0,work.points[i].flags.y!=0u);
    for(var j=0u;j<unique;j++) { rank -= params.values.z*length(work.points[i].x.xyz-work.points[j].x.xyz); }
    work.points[i].anchorB.w=rank;
  }
  for(var i=1u;i<unique;i++) { var j=i; loop { if(j==0u) { break; } if(work.points[j].anchorB.w>=work.points[j-1u].anchorB.w) { break; }
    let point=work.points[j]; work.points[j]=work.points[j-1u]; work.points[j-1u]=point; j--; } }
  if(!meshPlaneContacts(pair)) { _ = meshMeshContacts(pair); }
  results[slot]=work;
}
`;
export const sdfNarrowphaseWGSL = sdfNarrowphaseSource();
