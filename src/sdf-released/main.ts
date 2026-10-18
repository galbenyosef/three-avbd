import * as THREE from 'three/webgpu';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { BodyMesh, recommendedLimits } from '../lib/index.ts';
import { instantiateReleased, loadReleasedScene, type ReleasedProfile, type ReleasedScene, type Sample } from './scenes.ts';
import { readBrowserAsset } from './assets.ts';
import { measuredFrame } from './metrics.ts';
import '../sdf-showcase/style.css';
import './style.css';
const status=document.querySelector<HTMLOutputElement>('#status')!;
const select=document.querySelector<HTMLSelectElement>('#scene')!;
const profile=document.querySelector<HTMLSelectElement>('#profile')!;
const run=document.querySelector<HTMLButtonElement>('#run')!;
const reset=document.querySelector<HTMLButtonElement>('#reset')!;
const table=document.querySelector<HTMLTableSectionElement>('#metrics')!;
const format=(n:number)=>n===0?'0':Math.abs(n)<.001?n.toExponential(2):n.toLocaleString(undefined,{maximumFractionDigits:4});
async function main(){
  const renderer=new THREE.WebGPURenderer({antialias:true,requiredLimits:await recommendedLimits()});
  renderer.setPixelRatio(Math.min(devicePixelRatio,1.5));renderer.setSize(innerWidth,innerHeight);renderer.shadowMap.enabled=true;
  renderer.toneMapping=THREE.ACESFilmicToneMapping;document.body.append(renderer.domElement);
  const scene=new THREE.Scene();scene.background=new THREE.Color(0x101a22);scene.fog=new THREE.Fog(0x101a22,25,70);
  const camera=new THREE.PerspectiveCamera(45,innerWidth/innerHeight,.1,180);camera.position.set(9,7,12);
  const controls=new OrbitControls(camera,renderer.domElement);controls.target.set(0,2.5,0);controls.enableDamping=true;controls.minDistance=3;controls.maxDistance=40;controls.maxPolarAngle=Math.PI*.49;
  scene.add(new THREE.HemisphereLight(0xc4e1ff,0x34312b,2));
  const sun=new THREE.DirectionalLight(0xffe3bd,3);sun.position.set(4,14,8);sun.castShadow=true;sun.shadow.mapSize.set(1024,1024);sun.shadow.normalBias=.015;
  Object.assign(sun.shadow.camera,{left:-12,right:12,top:12,bottom:-12,far:45});scene.add(sun);
  let prepared:ReleasedScene|null=null,simulation:Awaited<ReturnType<typeof instantiateReleased>>|null=null;
  const group=new THREE.Group();scene.add(group);
  let paused=true,loading=false,inFlight=false,dirty=true,frame=0,last=0;
  controls.addEventListener('change',()=>{dirty=true;});
  const disposeMeshes=()=>{for(const object of [...group.children]){group.remove(object);const mesh=object as BodyMesh;mesh.geometry.dispose();mesh.material.dispose();}};
  function summary(){status.textContent=`${prepared?.capture.scene??''} · ${(frame/60).toFixed(2)} / 6.00 s · ${paused?'paused':'running'}`;run.textContent=frame===360?'Reset':paused?'Run':'Pause';}
  function metrics(actual:Sample){
    const reference=prepared!.capture.samples[actual.frame-1];
    const rows:[string,number,number][]=[['Contacts',actual.stats.contacts,reference.stats.contacts],['Pairs',actual.stats.pairs,reference.stats.pairs],['Paired queries',actual.stats.queries,reference.stats.queries],['Search iterations',actual.stats.iterations,reference.stats.iterations],['Kinetic energy',actual.kineticEnergy,reference.kineticEnergy],['Contact depth',actual.maxContactPenetration,reference.maxContactPenetration],['Lowest COM',actual.minCenterY,reference.minCenterY]];
    if(actual.maxVertexPenetration!==undefined)rows.push(['Sampled surface depth',actual.maxVertexPenetration,reference.maxVertexPenetration??0]);
    table.replaceChildren(...rows.map(([name,a,b])=>{const row=document.createElement('tr');for(const text of [name,format(a),format(b)]){const cell=document.createElement('td');cell.textContent=text;row.append(cell);}return row;}));
  }
  async function load(){
    if(loading)return;loading=true;paused=true;run.disabled=reset.disabled=select.disabled=profile.disabled=true;
    while(inFlight)await new Promise<void>(resolve=>requestAnimationFrame(()=>resolve()));
    status.textContent='Loading verified reference assets…';
    simulation?.world.destroy();simulation=null;disposeMeshes();prepared?.dispose();prepared=null;table.replaceChildren();
    try{
      prepared=await loadReleasedScene(Number(select.value),readBrowserAsset);
      simulation=await instantiateReleased(prepared,{renderer},profile.value as ReleasedProfile);
      const batches=new Map<THREE.BufferGeometry,number[]>();
      prepared.bodies.forEach((b,i)=>{const list=batches.get(b.geometry)??[];list.push(i);batches.set(b.geometry,list);});
      for(const [geometry,ids] of batches){
        const fixed=prepared.bodies[ids[0]].original.static;
        const color=fixed?0x243744:select.value==='1'?0xe3ad60:select.value==='6'?0x68c2bc:select.value==='9'?0xc2a1d9:0x8e9fe0;
        const mesh=new BodyMesh(simulation.world,{geometry,bodies:ids.map(i=>simulation!.bodies[i]),material:new THREE.MeshStandardNodeMaterial({color,roughness:.4,metalness:.12})});
        mesh.castShadow=!fixed;mesh.receiveShadow=true;group.add(mesh);
      }
      const distance=select.value==='0'?26:select.value==='1'?17:19;
      camera.position.copy(new THREE.Vector3(9,5,12).normalize().multiplyScalar(distance).add(controls.target));
      camera.setViewOffset(innerWidth,innerHeight,-Math.min(350,innerWidth*.4)/2,0,innerWidth,innerHeight);
      frame=0;summary();dirty=true;
    }catch(error){status.textContent=String(error);console.error(error);}
    finally{loading=false;select.disabled=profile.disabled=false;run.disabled=reset.disabled=!simulation;}
  }
  profile.addEventListener('change',()=>void load());select.addEventListener('change',()=>void load());reset.addEventListener('click',()=>void load());
  run.addEventListener('click',()=>{if(frame===360){void load();return;}paused=!paused;summary();});
  await load();
  renderer.setAnimationLoop((time:number)=>{
    if(document.hidden||time-last<1000/30)return;last=time;controls.update();
    if(!paused&&!loading&&!inFlight&&simulation&&prepared){
      inFlight=true;
      void measuredFrame(prepared,simulation.world,simulation.bodies,frame+1,frame+1===360).then(sample=>{
        frame=sample.frame;if(frame>=360)paused=true;metrics(sample);summary();dirty=true;
      }).catch(error=>{paused=true;status.textContent=String(error);console.error(error);}).finally(()=>{inFlight=false;});
    }
    if(dirty&&!loading){renderer.render(scene,camera);dirty=false;}
  });
  addEventListener('resize',()=>{camera.aspect=innerWidth/innerHeight;camera.setViewOffset(innerWidth,innerHeight,-Math.min(350,innerWidth*.4)/2,0,innerWidth,innerHeight);camera.updateProjectionMatrix();renderer.setSize(innerWidth,innerHeight);dirty=true;});
}
void main().catch(error=>{status.textContent=String(error);console.error(error);});
