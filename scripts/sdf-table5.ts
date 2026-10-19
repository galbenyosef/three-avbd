// Paper Table 5 (Giles & Andrews 2026, p. 10), Nelder–Mead rows: per-collision-call contacts,
// optimizer iterations and SDF queries on the operator-distorted Falling Boxes scenes.
// Compares this GPU narrowphase (released parameters) with the upstream CPU captures and the
// paper. Collision time is not measured. Bounded and sequential; no timing claims.
import { readFile, writeFile } from 'node:fs/promises';
import { create, globals } from 'webgpu';
import { loadReleasedScene, instantiateReleased, type ReadAsset } from '../src/sdf-released/scenes.ts';
import { measuredFrame } from '../src/sdf-released/metrics.ts';
const output=process.argv[2],frames=Number(process.argv[3]??360);
if(!output||!Number.isInteger(frames)||frames<1||frames>360)throw new Error('Usage: sdf-table5.ts output.json [frames 1..360]');
// Paper Table 5, NM rows: contacts, iterations, SDF queries per time step.
const paper:Record<string,[number,number,number]>={
  'Undistorted':[45.7,1066,2302],'BEND 0.15':[46.0,2389,5010],'BEND 0.55':[39.8,2346,4896],'BEND 0.95':[36.4,2842,5870],
  'DISPLACE 0.05':[42.0,1707,3600],'DISPLACE 0.10':[36.1,1861,3939],'DISPLACE 0.20':[36.1,2330,4982],
  'TWIST 0.50':[45.8,3161,6561],'TWIST 2.50':[42.1,3354,7081],'TWIST 5.00':[43.5,2743,5871]};
const configs:[string,number,string|undefined][]=[['Undistorted',2,undefined],
  ...['0.15','0.55','0.95'].map(k=>[`BEND ${k}`,3,k] as [string,number,string]),
  ...['0.05','0.10','0.20'].map(k=>[`DISPLACE ${k}`,5,k] as [string,number,string]),
  ...['0.50','2.50','5.00'].map(k=>[`TWIST ${k}`,4,k] as [string,number,string])];
const read:ReadAsset=async path=>{const b=await readFile(new URL(`../tests/fixtures/${path}`,import.meta.url));return b.buffer.slice(b.byteOffset,b.byteOffset+b.byteLength) as ArrayBuffer;};
Object.assign(globalThis,globals);
const gpu=create([]),adapter=await gpu.requestAdapter();if(!adapter)throw new Error('No WebGPU adapter');
const device=await adapter.requestDevice({requiredLimits:{maxStorageBuffersPerShaderStage:adapter.limits.maxStorageBuffersPerShaderStage}});
(globalThis as typeof globalThis & {__table5Gpu?:unknown}).__table5Gpu=[gpu,adapter,device];
const errors:string[]=[];device.onuncapturederror=e=>errors.push(e.error.message);
const mean=(values:number[])=>values.reduce((s,v)=>s+v,0)/values.length;
const rows:unknown[]=[];
try{
  for(const [name,id,k] of configs){
    const scene=await loadReleasedScene(id,read,k);
    const {world,bodies}=await instantiateReleased(scene,{device},'reference');
    const substeps=Math.round(scene.capture.solver.dt/world.dt);
    try{
      const ours:{contacts:number;pairs:number;iterations:number;queries:number}[]=[];
      for(let frame=1;frame<=frames;frame++){
        const s=await measuredFrame(scene,world,bodies,frame,false);
        ours.push({contacts:s.stats.contacts,pairs:s.stats.pairs,iterations:s.stats.iterations/substeps,queries:s.stats.queries/substeps});
        await device.queue.onSubmittedWorkDone();await new Promise(resolve=>setTimeout(resolve,5));
      }
      // Upstream captures sum iterations/queries over a frame's substeps; contacts are the last substep's.
      const ref=scene.capture.samples.slice(0,frames).map(s=>({contacts:s.stats.contacts,pairs:s.stats.pairs,iterations:s.stats.iterations/substeps,queries:s.stats.queries/substeps}));
      // Iterations per pair separates per-pair optimizer cost from trajectory-dependent pair counts.
      const avg=(list:typeof ours)=>({contacts:mean(list.map(x=>x.contacts)),pairs:mean(list.map(x=>x.pairs)),iterations:mean(list.map(x=>x.iterations)),queries:mean(list.map(x=>x.queries)),
        iterationsPerPair:mean(list.map(x=>x.iterations))/Math.max(mean(list.map(x=>x.pairs)),1e-9)});
      const row={name,scene:id,warpK:k??null,substeps,gpu:avg(ours),upstream:avg(ref),paper:{contacts:paper[name][0],iterations:paper[name][1],queries:paper[name][2]}};
      rows.push(row);
      console.log(name.padEnd(14),['gpu','upstream','paper'].map(key=>{const r=(row as any)[key];return `${key} ${r.contacts.toFixed(1)}/${Math.round(r.iterations)}/${Math.round(r.queries)}${r.pairs===undefined?'':` pairs ${r.pairs.toFixed(1)} it/pair ${r.iterationsPerPair.toFixed(1)}`}`;}).join(' | '));
    } finally {world.destroy();scene.dispose();}
  }
  if(errors.length)throw new Error(errors.join('\n'));
  await writeFile(output,JSON.stringify({paper:'Giles & Andrews 2026, Table 5 (NM rows)',reference:'sdf-dcd-demo 1dcdf1e7fa07d16fda15826ad8c2e6697075a76e',frames,profile:'reference (released parameters)',
    conventions:'per collision call: iterations and queries are frame sums divided by substeps; contacts are the last substep of each frame; averaged over frames',rows},null,2)+'\n');
} finally {device.destroy();}
