// Released scene inputs are immutable oracle captures, not regenerated goldens.
import * as THREE from 'three';
import { World, type Body, type Quat, type Vec3, type WorldOptions } from '../lib/world.ts';
import { AnalyticField, VoxelField, curvatureSeeds, principalInertia, type Field, type CollisionParams } from '../lib/sdf.ts';

export const releasedScenes = [{id:0,name:'Box Stack'}, {id:1,name:'Gear Stack'}, {id:6,name:'Pile of Chairs'}, {id:9,name:'Bunny Pile'}] as const;
export type ReadAsset = (path: string) => Promise<ArrayBuffer>;
export interface RecordedBody {
  position: Vec3; origin: Vec3; orientation: Quat; scale: number; static: boolean;
  velocity: Vec3; angularVelocity: Vec3; mass: number; inertia: number[][]; friction: number;
  shape: {kind: 'box' | 'voxel'; asset?: string; halfExtents?: Vec3; centerOfMass: Vec3; bounds: {min:Vec3;max:Vec3}; convex:boolean};
}
export interface Sample {
  frame: number; stats: {pairs:number;contacts:number;queries:number;iterations:number};
  kineticEnergy:number; maxContactPenetration:number; minCenterY:number;
  maxVertexPenetration?:number; vertexQueries?:number;
  penetrationWitness?:{a:number;b:number;point:Vec3;contacts:{point:number[];separation:number;normal:number[];cached:boolean}[]};
}
export interface Capture {
  scene:string; sceneIndex:number; frames:number;
  solver: {dt:number;substeps:number;iterations:number;gravity:Vec3;type:string};
  params:CollisionParams; initialBodies:RecordedBody[]; samples:Sample[];
}
export interface PreparedBody {
  original: RecordedBody; field: Field; geometry: THREE.BufferGeometry;
  rotation: Quat; fieldRotation: Quat; moments:Vec3;
}
export interface ReleasedScene { capture:Capture; bodies:PreparedBody[]; dispose():void }
const json = async (read:ReadAsset,path:string) => JSON.parse(new TextDecoder().decode(await read(path)));
const q = (v:Quat) => new THREE.Quaternion(...v).normalize();

async function binary(read:ReadAsset, folder:string, name:string):Promise<Float32Array> {
  const manifest=await json(read,`${folder}/manifest.json`);
  const compressed=await read(`${folder}/${name}.bin.gz`);
  // Some static servers transparently decode .gz via Content-Encoding. Hash either path.
  const header=new Uint8Array(compressed,0,Math.min(2,compressed.byteLength));
  const data=header[0]===0x1f&&header[1]===0x8b
    ? await new Response(new Blob([compressed]).stream().pipeThrough(new DecompressionStream('gzip'))).arrayBuffer()
    : compressed;
  const expected=(manifest.grids??manifest.surfaces)[name];
  const hash=Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',data)),v=>v.toString(16).padStart(2,'0')).join('');
  if(data.byteLength!==expected.bytes || hash!==expected.sha256)throw new Error(`Oracle asset checksum mismatch: ${folder}/${name}`);
  // Explicit little-endian decode keeps the loader independent of host byte order.
  const view=new DataView(data), floats=new Float32Array(data.byteLength/4);
  for(let i=0;i<floats.length;i++)floats[i]=view.getFloat32(i*4,true);
  return floats;
}

export async function loadReleasedScene(id:number,read:ReadAsset):Promise<ReleasedScene> {
  if(!releasedScenes.some(scene=>scene.id===id))throw new Error('Unknown released scene');
  const capture:Capture=await json(read,`oracle-sdf/scene-${id}.json`);
  const fields=new Map<string,{field:Field;geometry:THREE.BufferGeometry}>();
  const meshes=new Map<string,THREE.BufferGeometry>();
  const bodies:PreparedBody[]=[];
  try {
    for(const original of capture.initialBodies) {
      const key=JSON.stringify(original.shape);
      let shape=fields.get(key);
      if(!shape) {
        if(original.shape.kind==='box') {
          const half=original.shape.halfExtents!;
          const field=new AnalyticField({kind:'box',halfExtents:half});
          field.centerOfMass=[...original.shape.centerOfMass];
          // The released Ground is deliberately not finalized (no curvature seeds).
          field.seeds=original.static?[]:curvatureSeeds(field);
          shape={field,geometry:new THREE.BoxGeometry(2*half[0],2*half[1],2*half[2]).toNonIndexed()};
        } else {
          const name=original.shape.asset!.toLowerCase();
          const metadata=await json(read,`oracle-sdf/${name}.json`);
          const seed=await json(read,`oracle-sdf/seeds-${name}.json`);
          const grid=await binary(read,'oracle-sdf-grids',name);
          const surface=await binary(read,'oracle-sdf-surfaces',name);
          const field=new VoxelField(metadata.grid.dimensions,metadata.grid.origin,metadata.grid.cell,grid,metadata.bounds,surface);
          field.centerOfMass=[...seed.centerOfMass] as Vec3; field.seeds=seed.seeds.map((p:{p:Vec3})=>[...p.p]);
          const geometry=new THREE.BufferGeometry(); geometry.setAttribute('position',new THREE.BufferAttribute(surface,3)); geometry.computeVertexNormals();
          shape={field,geometry};
        }
        fields.set(key,shape);
      }
      const principal=original.static ? {moments:[1,1,1] as Vec3,rotation:[0,0,0,1] as Quat} : principalInertia(original.inertia.flat());
      const fieldRotation=q(principal.rotation).invert().toArray() as Quat;
      const rotation=q(original.orientation).multiply(q(principal.rotation)).normalize().toArray() as Quat;
      const meshKey=JSON.stringify([key,original.scale,fieldRotation]);
      let geometry=meshes.get(meshKey);
      if(!geometry) {
        geometry=shape.geometry.clone();
        const com=shape.field.centerOfMass;
        geometry.translate(-com[0],-com[1],-com[2]); geometry.scale(original.scale,original.scale,original.scale);
        geometry.applyQuaternion(q(fieldRotation)); meshes.set(meshKey,geometry);
      }
      bodies.push({original,field:shape.field,geometry,rotation,fieldRotation,moments:principal.moments});
    }
    return {capture,bodies,dispose:()=>{for(const geometry of meshes.values())geometry.dispose();}};
  } catch(error) {for(const geometry of meshes.values())geometry.dispose();throw error;}
  finally {for(const shape of fields.values())shape.geometry.dispose();}
}

export type ReleasedProfile = 'reference' | 'mesh-quality';
/** Quality settings are explicit; immutable released inputs remain available for comparisons. */
export function releasedSettings(scene:ReleasedScene,profile:ReleasedProfile='mesh-quality') {
  const mesh=scene.bodies.some(body=>body.original.shape.kind==='voxel');
  const quality=profile==='mesh-quality'&&mesh;
  return {substeps:scene.capture.solver.substeps,collision:{...scene.capture.params,...(quality?{
    signedSeparation:true,refreshDeeperContacts:true,meshVertexContacts:true,margin:.04,allowedPenetration:.04-(scene.capture.params.margin-scene.capture.params.allowedPenetration),
  }:{})}};
}

export async function instantiateReleased(scene:ReleasedScene, options:Omit<WorldOptions,'maxBodies'|'dt'|'iterations'|'gravity'>,profile:ReleasedProfile='mesh-quality'):Promise<{world:World;bodies:Body[]}> {
  const source=scene.capture.solver,settings=releasedSettings(scene,profile);
  const world=await World.create({...options,maxBodies:scene.bodies.length,dt:source.dt/settings.substeps,iterations:source.iterations,gravity:source.gravity});
  world.maxSubsteps=settings.substeps;world.readbackEvery=0;
  try {
    const bodies=scene.bodies.map(({original:b,rotation,moments})=>world.addBox({
      size:[1,1,1],position:b.position,rotation,velocity:b.velocity,angularVelocity:b.angularVelocity,
      fixed:b.static,friction:b.friction,...(!b.static?{massProperties:{mass:b.mass,moments}}:{}),
    })!);
    await world.enableSdfCollisions(scene.bodies.map((b,i)=>({body:bodies[i],field:b.field,scale:b.original.scale,rotation:b.fieldRotation})),settings.collision,{maxPairs:bodies.length*(bodies.length-1)/2,autoGrow:false});
    return {world,bodies};
  } catch(error) {world.destroy();throw error;}
}
