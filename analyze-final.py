import json,math,sys
from pathlib import Path
root=Path(sys.argv[1]); prior=Path(sys.argv[2])
def read(p):return json.loads(p.read_text())
def compare(ref,rows):
 a=[v for r in ref for v in r['logprobs']];b=[v for r in rows for v in r['logprobs']]
 assert len(a)==len(b)==240
 maximum=0.;flips=[]
 for i,(x,y) in enumerate(zip(a,b)):
  assert x.keys()==y.keys() and len(x)==512
  assert all(math.isfinite(v) for v in [*x.values(),*y.values()])
  maximum=max(maximum,max(abs(x[k]-y[k]) for k in x))
  tx,ty=max(x,key=x.get),max(y,key=y.get)
  if tx!=ty:
   margin=max(x[tx]-x[ty],y[ty]-y[tx]);assert margin<=0.005
   flips.append({'index':i,'margin':margin})
 assert maximum<=0.005,maximum
 return {'max_abs_logprob_delta':maximum,'argmax_changes':flips,'hit_requests':sum(r['cached']>0 for r in rows)}
def metric(d,phase,name):return sum(m.get('value',0) for m in d[phase] if m['name']==name)
report={}
for label in ['fix','feature']:
 observed=read(root/(label+'-verified.json'))
 assert observed==read(root/'source-manifest.json')[label]
for name in ['fix-cpu','feature-cpu','full-base']:
 assert (root/(name+'.exit')).read_text().strip()=='0',name
for family in ['full','replica','sharded']:
 ref=read(root/'full-base.json')['cold'] if family=='full' else read(prior/('replica-base.json' if family=='replica' else 'teacher-base-tp1-r1.json'))['cold']
 for tp in [1,2]:
  for role in ['producer','read']:
   assert (root/(f'{family}-{role}-tp{tp}.exit')).read_text().strip()=='0'
  name=f'{family}-read-tp{tp}';d=read(root/(name+'.json'))
  assert all(w['canonical_layout'] and w['all_layers_portable'] for w in d['workers'])
  expected={'FullAttentionSpec'} if family=='full' else {'FullAttentionSpec','ChunkedLocalAttentionSpec'}
  assert all(set(w['specs'])==expected for w in d['workers'])
  fs=metric(d,'metrics_before','vllm:kv_offload_tiering_read_bytes')
  before=metric(d,'metrics_before','vllm:kv_offload_load_bytes');after=metric(d,'metrics_after','vllm:kv_offload_load_bytes')
  assert fs>0 and after>before>0
  item={'commit':read(root/'source-manifest.json')['fix' if family=='full' else 'feature']['commit'],'fs_read_bytes':fs,'load_before':before,'load_after':after,'phases':{}}
  for phase in ['cold','reload']:
   item['phases'][phase]=compare(ref,d[phase]);assert item['phases'][phase]['hit_requests']==240
  assert 'block I/O failed' not in (root/(name+'.log')).read_text()
  report[name]=item
for name in ['stress-tp2','read-after-stress-tp1']:
 assert (root/(name+'.exit')).read_text().strip()=='0'
 d=read(root/(name+'.json'));ref=read(prior/'replica-base.json')['cold']
 item={phase:compare(ref,d[phase]) for phase in ['cold','reload']}
 assert metric(d,'metrics_before','vllm:kv_offload_tiering_read_bytes')>0
 assert metric(d,'metrics_after','vllm:kv_offload_load_bytes')>metric(d,'metrics_before','vllm:kv_offload_load_bytes')>0
 if name=='stress-tp2':
  assert len(d['stress'])==6
  assert all(r['fs_read_bytes']>0 and r['load_bytes']>0 and r['max_logprob_delta']<=0.005 for r in d['stress'])
  item['cycles']=d['stress']
 else: assert all(item[p]['hit_requests']==240 for p in ['cold','reload'])
 assert 'block I/O failed' not in (root/(name+'.log')).read_text()
 report[name]=item
assert (root/'run.exit').read_text().strip()=='0'
print(json.dumps(report,indent=2))
