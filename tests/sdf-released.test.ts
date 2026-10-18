import assert from 'node:assert/strict';
import { test } from 'node:test';
import { gunzipSync } from 'node:zlib';
import { readFile } from 'node:fs/promises';
import { Matrix4, Quaternion, Vector3 } from 'three';
import { principalInertia } from '../src/lib/inertia.ts';
import { loadReleasedScene, releasedSettings, releasedScenes, type ReadAsset } from '../src/sdf-released/scenes.ts';
const read:ReadAsset=async path=>{const b=await readFile(new URL(`./fixtures/${path}`,import.meta.url));return b.buffer.slice(b.byteOffset,b.byteOffset+b.byteLength) as ArrayBuffer;};
const near=(a:number,b:number)=>assert.ok(Math.abs(a-b)<1e-6*Math.max(1,Math.abs(b)),`${a} != ${b}`);

test('principal inertia rejects invalid tensors and reconstructs off-diagonal inertia',()=>{
  for(const tensor of [[0,0,0,0,0,0,0,0,0],[1,2,0,0,1,0,0,0,1],[1,0,0,0,-1,0,0,0,1]])assert.throws(()=>principalInertia(tensor));
  const tensor=[2,.3,.2,.3,3,.4,.2,.4,4],p=principalInertia(tensor);
  const m=new Matrix4().makeRotationFromQuaternion(new Quaternion(...p.rotation)).elements;
  for(let i=0;i<3;i++)for(let j=0;j<3;j++)near(p.moments.reduce((sum,v,k)=>sum+v*m[k*4+i]*m[k*4+j],0),tensor[i*3+j]);
});

for(const entry of releasedScenes)test(`released ${entry.name} preserves full inertia, source poses and collision/render alignment`,async()=>{
  const scene=await loadReleasedScene(entry.id,read);
  try{
    assert.deepEqual(releasedSettings(scene,'reference').collision,scene.capture.params);
    const quality=releasedSettings(scene).collision;
    assert.equal(quality.meshPlaneContacts,entry.id===0?undefined:true,'mesh vertex contacts are enabled for mesh piles only');
    assert.equal(releasedSettings(scene,'reference').collision.meshPlaneContacts,undefined);
    near(quality.margin-quality.allowedPenetration,scene.capture.params.margin-scene.capture.params.allowedPenetration);
    assert.equal(quality.margin,entry.id===0?scene.capture.params.margin:.04);
    assert.equal(quality.signedSeparation,entry.id===0?undefined:true);
    assert.equal(releasedSettings(scene,'reference').collision.signedSeparation,undefined);
    assert.equal(scene.capture.params.cacheSize,4,'quality never mutates the captured parameters');
    assert.equal(scene.bodies.length,entry.id===0?29:entry.id===1?11:28);
    for(const body of scene.bodies){
      const b=body.original;
      if(b.static)assert.equal(body.field.seeds.length,0,'released Ground is not finalized');
      const fieldQ=new Quaternion(...body.fieldRotation),bodyQ=new Quaternion(...body.rotation),sourceQ=new Quaternion(...b.orientation).normalize();
      near(Math.abs(bodyQ.clone().multiply(fieldQ).dot(sourceQ)),1);
      if(!b.static){
        const matrix=new Matrix4().makeRotationFromQuaternion(fieldQ.clone().invert()).elements;
        for(let i=0;i<3;i++)for(let j=0;j<3;j++)near(body.moments.reduce((sum,v,k)=>sum+v*matrix[k*4+i]*matrix[k*4+j],0),b.inertia[i][j]);
      }
      // A render vertex transformed back to the original source frame must lie on its field.
      const p=body.geometry.getAttribute('position');
      for(let i=0;i<p.count;i+=Math.max(1,Math.floor(p.count/30))){
        const source=new Vector3().fromBufferAttribute(p,i).applyQuaternion(fieldQ.clone().invert()).multiplyScalar(1/b.scale).add(new Vector3(...body.field.centerOfMass));
        assert.ok(Math.abs(body.field.sample(source.toArray()).phi)<.025,'source surface within voxel interpolation tolerance');
      }
    }
  } finally {scene.dispose();}
});


test('released assets also accept HTTP-decoded gzip and still verify content hashes',async()=>{
  const decoded:ReadAsset=async path=>{
    const data=await read(path);if(!path.endsWith('.gz'))return data;
    const raw=gunzipSync(new Uint8Array(data));return raw.buffer.slice(raw.byteOffset,raw.byteOffset+raw.byteLength) as ArrayBuffer;
  };
  const scene=await loadReleasedScene(1,decoded);assert.equal(scene.bodies.length,11);scene.dispose();
  await assert.rejects(loadReleasedScene(1,async path=>path.endsWith('.gz')?new ArrayBuffer(4):read(path)),/checksum mismatch/);
});
