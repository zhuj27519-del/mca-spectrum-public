"""Reproducible five-class prototype. No hardware or real-world claims."""
from pathlib import Path
import os,json,argparse,hashlib,time,platform,sys
ROOT=Path(__file__).resolve().parents[1]
os.environ['MPLCONFIGDIR']=str(ROOT/'work/mpl-cache')
os.environ['OMP_NUM_THREADS']='4'; os.environ['MKL_NUM_THREADS']='4'
import numpy as np
import torch
from torch import nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
CLASSES=['Cs-137','Co-60','Ba-133','Am-241','Background']; RN=['Cs137','Co60','Ba133','Am241']
torch.set_num_threads(4)
def savejson(p,obj): p.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf8')
def getdata(run,split): return dict(np.load(run/'data'/f'{split}.npz'))
def crop(a): return a[20:2068].reshape(1024,2).sum(1)
def generate(run,small=False):
 # INPUTS AND LABELS: original response shapes; no upstream tail edits or Am replacement.
 cfg=dict(seed=20261116,classes=CLASSES,channels=1024,energy_lower_keV=20,bin_width_keV=2,expected_counts=[200,50000],background_fraction=[.1,.95],scope_mm=[.001,4],preprocessing='32*sqrt(counts/sum); no fitted statistics',source_commit='444156e4befd3754b101848a16e862e502fbcfd3',small=small)
 rng=np.random.default_rng(cfg['seed']); order=rng.permutation(30)
 conditions={'train':sorted(order[:18].tolist()),'val':sorted(order[18:24].tolist()),'test_same':sorted(order[:18].tolist()),'test_held':sorted(order[24:].tolist())}
 cfg['conditions']=conditions; cfg['sizes']={'train':500 if small else 6000,'val':150 if small else 1500,'test_same':150 if small else 1500,'test_held':150 if small else 1500}
 (run/'data').mkdir(parents=True,exist_ok=True); (run/'models').mkdir(exist_ok=True); (run/'figures').mkdir(exist_ok=True)
 responses=[]; sources=[]; thick=[]
 for n in RN:
  files=sorted((ROOT/'upstream/data/Simulation_steel_sphere'/n).glob('*.dat'),key=lambda p:float(p.name.split('STEEL')[1].replace('mm.dat','').replace('__','.')))[:30]
  bank=[]
  for p in files:
   a=crop(np.loadtxt(p)); assert np.isfinite(a).all() and a.min()>=0 and a.sum()>0
   bank.append(a/a.sum()); sources.append(dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
  responses.append(bank)
  if not thick: thick=[float(p.name.split('STEEL')[1].replace('mm.dat','').replace('__','.')) for p in files]
 bpath=next((ROOT/'upstream/data').glob('SPS*')); b=crop(np.loadtxt(bpath)); b=b/b.sum(); sources.append(dict(path=str(bpath.relative_to(ROOT)),sha256=hashlib.sha256(bpath.read_bytes()).hexdigest()))
 bank=np.array(responses); cfg['thickness_mm']=thick; cfg['sources']=sources
 # Whole normalized response hashes must be disjoint across train / val / held conditions.
 hashes={s:{hashlib.sha256(bank[k,i].tobytes()).hexdigest() for k in range(4) for i in conditions[s]} for s in ['train','val','test_held']}
 assert not(hashes['train']&hashes['val'] or hashes['train']&hashes['test_held'] or hashes['val']&hashes['test_held'])
 seen=set(); checks={}
 for split,n in cfg['sizes'].items():
  srng=np.random.default_rng(np.random.SeedSequence([cfg['seed'],list(cfg['sizes']).index(split)+1]))
  labels=np.arange(n)%5; srng.shuffle(labels); counts=[]; expected=[]; fractions=[]; ids=[]
  for label in labels:
   ci=int(srng.choice(conditions[split])) if label<4 else -1
   total=float(np.exp(srng.uniform(np.log(200),np.log(50000)))); f=float(srng.uniform(.1,.95)) if label<4 else 1.
   p=f*b+(1-f)*bank[label,ci] if label<4 else b
   y=srng.poisson(total*p).astype(np.int32); assert y.sum()>0
   h=hashlib.sha256(y.tobytes()).hexdigest(); assert h not in seen,'Exact spectrum duplicate'; seen.add(h)
   counts.append(y); expected.append(total); fractions.append(f); ids.append(ci)
  np.savez_compressed(run/'data'/f'{split}.npz',counts=np.array(counts),labels=labels.astype(np.int64),expected_total=np.array(expected),background_fraction=np.array(fractions),condition_index=np.array(ids),sample_id=np.array([f'{split}_{i:05d}' for i in range(n)]))
  checks[split]=dict(samples=n,class_counts=np.bincount(labels,minlength=5).tolist(),exact_duplicates=0)
 np.savez_compressed(run/'data/response_bank.npz',responses=bank,background=b)
 savejson(run/'config.json',cfg); savejson(run/'data_audit.json',dict(splits=checks,response_hash_overlap=False,background_limitation='All splits share one measured background shape. Held conditions apply only to four source classes.',source_conditions='0.001-4 mm only; no extrapolation beyond this range tested.'))
 print('Generated',cfg['sizes'],flush=True)
def preprocess(x):
 # No fitted mean, variance or test statistics.
 return (32*np.sqrt(x/np.maximum(x.sum(1,keepdims=True),1))).astype(np.float32)[:,None,:]
class SmallCNN(nn.Module):
 def __init__(self):
  super().__init__(); self.net=nn.Sequential(nn.Conv1d(1,8,9,stride=2,padding=4),nn.ReLU(),nn.MaxPool1d(2),nn.Conv1d(8,16,7,stride=2,padding=3),nn.ReLU(),nn.Conv1d(16,24,5,stride=2,padding=2),nn.ReLU(),nn.AdaptiveAvgPool1d(32),nn.Flatten(),nn.Linear(24*32,48),nn.ReLU(),nn.Linear(48,5))
 def forward(self,x): return self.net(x)
def baseline_scores(x,templates,labels,penalty=0):
 # Full-spectrum multinomial log likelihood, with train response + background grid.
 logp=np.log(np.maximum(templates,1e-12)).T; pieces=[]
 for start in range(0,len(x),128):
  a=x[start:start+128]; likelihood=(a/np.maximum(a.sum(1,keepdims=True),1))@logp
  s=np.stack([likelihood[:,labels==k].max(1) for k in range(5)],axis=1); s[:,:4]-=penalty; pieces.append(s)
 return np.concatenate(pieces)
def cnn_scores(model,x):
 model.eval(); result=[]
 with torch.inference_mode():
  for i in range(0,len(x),128): result.append(torch.softmax(model(torch.from_numpy(preprocess(x[i:i+128]))),1).numpy())
 return np.concatenate(result)
def train(run,epochs):
 cfg=json.loads((run/'config.json').read_text()); torch.manual_seed(cfg['seed']); np.random.seed(cfg['seed']); torch.use_deterministic_algorithms(True)
 trainset=getdata(run,'train'); val=getdata(run,'val'); banks=np.load(run/'data/response_bank.npz'); templates=[]; labels=[]
 for k in range(4):
  for i in cfg['conditions']['train']:
   for f in [.1,.25,.4,.55,.7,.85,.95]: templates.append((1-f)*banks['responses'][k,i]+f*banks['background']); labels.append(k)
 templates.append(banks['background']); labels.append(4); templates=np.array(templates); labels=np.array(labels)
 raw=baseline_scores(val['counts'],templates,labels); choices=[]
 for penalty in [0,.001,.003,.01,.03]:
  s=raw.copy(); s[:,:4]-=penalty; choices.append((float((s.argmax(1)==val['labels']).mean()),penalty))
 accuracy,penalty=max(choices,key=lambda t:(t[0],-t[1])); np.savez_compressed(run/'models/baseline.npz',templates=templates,labels=labels,penalty=penalty)
 model=SmallCNN(); optimizer=torch.optim.Adam(model.parameters(),lr=.001); loss_fn=nn.CrossEntropyLoss()
 ds=torch.utils.data.TensorDataset(torch.from_numpy(preprocess(trainset['counts'])),torch.from_numpy(trainset['labels']))
 loader=torch.utils.data.DataLoader(ds,batch_size=128,shuffle=True,num_workers=0,generator=torch.Generator().manual_seed(cfg['seed']))
 hist=[]; best=-1; t0=time.perf_counter()
 for epoch in range(epochs):
  model.train(); total=0
  for x,y in loader:
   optimizer.zero_grad(); loss=loss_fn(model(x),y); loss.backward(); optimizer.step(); total+=loss.item()*len(y)
  scores=cnn_scores(model,val['counts']); acc=float((scores.argmax(1)==val['labels']).mean()); vl=float(-np.log(np.maximum(scores[np.arange(len(scores)),val['labels']],1e-12)).mean())
  hist.append(dict(epoch=epoch+1,train_loss=total/len(ds),val_loss=vl,val_accuracy=acc))
  if acc>best: best=acc; bestepoch=epoch+1; torch.save(model.state_dict(),run/'models/cnn.pt')
  print(f'Epoch {epoch+1}/{epochs}: loss {total/len(ds):.4f}, validation {acc:.4f}',flush=True)
 savejson(run/'history.json',hist)
 savejson(run/'models/metadata.json',dict(classes=CLASSES,preprocess=cfg['preprocessing'],cnn_parameters=sum(p.numel() for p in model.parameters()),selected_epoch=bestepoch,cnn_val_accuracy=best,baseline_val_accuracy=accuracy,baseline_penalty=penalty,baseline_candidates=choices,selection='Validation accuracy, earliest epoch on tie; tests untouched during selection.',train_seconds=time.perf_counter()-t0,epochs=epochs,device='cpu',threads=4,torch=torch.__version__,numpy=np.__version__,python=sys.version,processor=platform.processor()))
 fig,ax=plt.subplots(1,2,figsize=(10,4),layout='constrained'); ax[0].plot([h['train_loss'] for h in hist],label='train'); ax[0].plot([h['val_loss'] for h in hist],label='validation'); ax[0].legend(); ax[0].set(title='Cross entropy',xlabel='Epoch (zero-based)'); ax[1].plot([h['val_accuracy'] for h in hist]); ax[1].set(title='Validation accuracy',xlabel='Epoch (zero-based)',ylim=(0,1)); fig.savefig(run/'figures/training.png',dpi=160); plt.close(fig)
 print('Model frozen; evaluation may now run.',flush=True)
def metrics(y,p):
 cm=np.zeros((5,5),int); np.add.at(cm,(y,p),1); rows=[]
 for k in range(5):
  tp=int(cm[k,k]); precision=tp/max(int(cm[:,k].sum()),1); recall=tp/max(int(cm[k].sum()),1)
  rows.append(dict(class_name=CLASSES[k],precision=precision,recall=recall,f1=2*precision*recall/max(precision+recall,1e-12),support=int(cm[k].sum())))
 return dict(n=len(y),accuracy=float((y==p).mean()) if len(y) else None,macro_f1=float(np.mean([a['f1'] for a in rows])),classes=rows,confusion=cm.tolist())
def evaluate(run):
 # Final tests are accessed only after checkpoint and baseline parameters are frozen.
 model=SmallCNN(); model.load_state_dict(torch.load(run/'models/cnn.pt',weights_only=True)); model.eval(); b=np.load(run/'models/baseline.npz'); results={}
 for split in ['test_same','test_held']:
  d=getdata(run,split); results[split]={}; scores={'baseline':baseline_scores(d['counts'],b['templates'],b['labels'],float(b['penalty'])),'cnn':cnn_scores(model,d['counts'])}
  for name,s in scores.items():
   p=s.argmax(1); result=metrics(d['labels'],p); result['by_count']={}; result['by_background']={}
   total=d['counts'].sum(1)
   for low,high in [(0,1000),(1000,10000),(10000,1000000)]:
    mask=(total>=low)&(total<high); result['by_count'][f'[{low},{high})']=metrics(d['labels'][mask],p[mask])
   for low,high in [(.1,.4),(.4,.7),(.7,.951)]:
    mask=(d['labels']<4)&(d['background_fraction']>=low)&(d['background_fraction']<high); result['by_background'][f'[{low},{high}) source classes only']=metrics(d['labels'][mask],p[mask])
   mask=d['labels']==4; result['background_only']=metrics(d['labels'][mask],p[mask]); results[split][name]=result
   fig,ax=plt.subplots(figsize=(7,6),layout='constrained'); cm=np.array(result['confusion']); ax.imshow(cm,cmap='Blues'); ax.set(xticks=range(5),yticks=range(5),xticklabels=CLASSES,yticklabels=CLASSES,xlabel='Predicted',ylabel='True',title=f'{name} | {split} | accuracy={result["accuracy"]:.3f}'); ax.tick_params(axis='x',rotation=30)
   for i in range(5):
    for j in range(5): ax.text(j,i,str(cm[i,j]),ha='center',va='center',color='white' if cm[i,j]>cm.max()/2 else 'black')
   fig.savefig(run/f'figures/{split}_{name}_confusion.png',dpi=150); plt.close(fig)
  np.savez_compressed(run/'data'/f'{split}_predictions.npz',**scores)
  bad=np.flatnonzero((scores['cnn'].argmax(1)!=d['labels'])|(scores['baseline'].argmax(1)!=d['labels']))[:6]
  fig,axes=plt.subplots(3,2,figsize=(12,10),layout='constrained')
  for ax,i in zip(axes.flat,bad):
   ax.plot(20+2*np.arange(1024),d['counts'][i],lw=.7); ax.set(title=f'{split}_{i:05d}: true {CLASSES[d["labels"][i]]}\nCNN {CLASSES[scores["cnn"][i].argmax()]} / baseline {CLASSES[scores["baseline"][i].argmax()]}',xlabel='Energy (keV)',ylabel='Counts')
  for ax in list(axes.flat)[len(bad):]: ax.axis('off')
  fig.savefig(run/f'figures/{split}_errors.png',dpi=150); plt.close(fig)
 sample=getdata(run,'test_same')['counts'][:1]; timing={}
 for name,fn in [('cnn',lambda:cnn_scores(model,sample)),('baseline',lambda:baseline_scores(sample,b['templates'],b['labels'],float(b['penalty'])))]:
  for _ in range(10): fn()
  times=[]
  for _ in range(100):
   start=time.perf_counter(); fn(); times.append((time.perf_counter()-start)*1000)
  timing[name]=dict(median_ms=float(np.median(times)),p95_ms=float(np.percentile(times,95)),file_bytes=(run/('models/cnn.pt' if name=='cnn' else 'models/baseline.npz')).stat().st_size)
 savejson(run/'evaluation.json',results); savejson(run/'timing.json',dict(measurement='CPU, 4 torch/OMP/MKL threads, batch=1, 10 warmups, 100 repetitions, includes preprocessing/scoring, excludes disk load/GUI, model resident; baseline includes log-template conversion.',models=timing))
 fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
 for name in ['baseline','cnn']:
  r=results['test_held'][name]; axes[0].plot(range(3),[v['accuracy'] for v in r['by_count'].values()],'-o',label=name); axes[1].plot(range(3),[v['accuracy'] for v in r['by_background'].values()],'-o',label=name)
 for ax in axes: ax.set(ylim=(0,1),ylabel='Accuracy'); ax.legend(); ax.grid(alpha=.2)
 axes[0].set(xticks=range(3),xticklabels=['<1000','1000-9999','>=10000'],title='Held conditions: realized total counts'); axes[1].set(xticks=range(3),xticklabels=['10-40%','40-70%','70-95%'],title='Held conditions: expected background, source only'); fig.savefig(run/'figures/stratified.png',dpi=160); plt.close(fig)
 print(json.dumps({s:{m:v['accuracy'] for m,v in a.items()} for s,a in results.items()}),flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser(); ap.add_argument('stage',choices=['generate','train','evaluate','all']); ap.add_argument('--run',default='runs/main'); ap.add_argument('--small',action='store_true'); ap.add_argument('--epochs',type=int,default=25); args=ap.parse_args(); run=ROOT/args.run; run.mkdir(parents=True,exist_ok=True)
 if args.stage in ('generate','all'): generate(run,args.small)
 if args.stage in ('train','all'): train(run,args.epochs)
 if args.stage in ('evaluate','all'): evaluate(run)
