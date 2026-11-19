"""Fixed public-template SYNTHETIC response coefficient experiment. No Bq/kg prediction."""
from pathlib import Path
import argparse,json,csv,hashlib,zipfile,shutil
import numpy as np
from scipy.optimize import nnls
import regression as r
from audit_next_data import csv_read
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
TARGETS=['K-template coefficient','U-template coefficient','Th-template coefficient']
WARNING='合成响应系数；固定响应下的新混合/新泊松噪声，不是实测活度或元素含量。'
CONFIG=dict(version='synthetic-response-v1',seed=20261117,parent_mixtures=600,replicates_per_parent=2,
    split_parents=[360,120,120],coefficients='independent Uniform(5,100) for each of 3 targets',
    live_times_s=[60,600],response='HF_Cs K/U/Th templates times bin width, summed to 64 intervals, each column normalized to sum 1',
    background='known synthetic constant 0.05 counts/s per rebinned interval; not measured environmental background',
    noise='independent Poisson(t * (R @ a + b)); conditional on fixed response; NOT MCNP statistical noise',
    input='gross synthetic counts / generated live time; train-only standardization; absolute amplitude retained',
    architecture=[64,8,3],epochs=500,learning_rate=.001,ridge_alpha=1.,
    selection='all hyperparameters fixed before generation; validation reported only, no model selection or test tuning',
    target_units='synthetic response coefficient (arbitrary declared scale); NO Bq/kg or elemental conversion',
    targets=TARGETS,warning=WARNING,source_doi='10.5281/zenodo.20032395',license='CC-BY-4.0',
    source_zip_sha256='63ccfd3bc72cc7f26dd4e880a137f8a717fbe9a23a1273af71edc955518ea280')

def response():
    p=ROOT/'data/next_search/mobile.zip';assert hashlib.sha256(p.read_bytes()).hexdigest()==CONFIG['source_zip_sha256']
    rows=[];source=[]
    with zipfile.ZipFile(p) as z:
        for name in ['K','U','Th']:
            n=f'data_v1_1_0/HF_Cs/Templates/{name}.csv';b=z.read(n);_,header,a=csv_read(b)
            assert a.shape[1]==2 and header[1]=='Counts_kg_per_s_per_keV_per_Bq'
            if rows:np.testing.assert_allclose(energy,a[:,0],rtol=0,atol=1e-9)
            energy=a[:,0];rows.append(a[:,1]);source.append(dict(file=n,sha256=hashlib.sha256(b).hexdigest(),original_header=header))
    width=np.median(np.diff(energy));np.testing.assert_allclose(np.diff(energy),width,atol=1e-9)
    fine=np.array(rows).T*width;groups=np.array_split(np.arange(len(energy)),64)
    R=np.array([fine[g].sum(axis=0) for g in groups]);sums=R.sum(axis=0);R/=sums
    assert np.linalg.matrix_rank(R)==3 and (R>=0).all()
    return R,np.array([energy[g].mean() for g in groups]),source,sums,[[int(g[0]),int(g[-1]),float(energy[g[0]]),float(energy[g[-1]])] for g in groups]
def metric(y,p):
    d=p-y
    return dict(MAE=np.abs(d).mean(axis=0).tolist(),RMSE=np.sqrt((d*d).mean(axis=0)).tolist(),bias=d.mean(axis=0).tolist())
def train(out):
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True);r.dump(out/'config.json',CONFIG)
    R,energy,source,sums,bins=response();r.dump(out/'response_source.json',dict(files=source,original_integrals=sums.tolist(),intervals=bins,condition_number=float(np.linalg.cond(R))))
    rng=np.random.default_rng(CONFIG['seed']);parents=rng.uniform(5,100,size=(600,3));perm=rng.permutation(600)
    group=np.repeat(np.arange(600),2);y=parents[group];live=np.tile([60.,600.],600);b=np.full(64,.05)
    rate=y@R.T+b;counts=rng.poisson(live[:,None]*rate);X=counts/live[:,None]
    split_groups=dict(train=perm[:360],validation=perm[360:480],test=perm[480:]);idx={k:np.flatnonzero(np.isin(group,g)) for k,g in split_groups.items()}
    assert not(set(split_groups['train'])&set(split_groups['validation']) or set(split_groups['train'])&set(split_groups['test']) or set(split_groups['test'])&set(split_groups['validation']))
    r.dump(out/'splits.json',{k:dict(parent_ids=split_groups[k].tolist(),row_indices=v.tolist()) for k,v in idx.items()})
    np.savez_compressed(out/'dataset.npz',counts=counts,live_time_s=live,X_rate=X,y_coefficients=y,parent_id=group,response=R,background=b,energy_keV=energy)
    tr,va,te=idx['train'],idx['validation'],idx['test'];s=r.scaling(X[tr],y[tr]);r.CONFIG.update(seed=CONFIG['seed'],epochs=500,learning_rate=.001,ridge_alpha=1.)
    coef=r.fit_ridge(X[tr],y[tr],s);p,history=r.fit_mlp(X[tr],y[tr],s)
    np.savez_compressed(out/'model.npz',**p,**s,ridge_coef=coef,train_target_mean=y[tr].mean(axis=0),response=R,background=b,targets=TARGETS,version=CONFIG['version'],warning=WARNING)
    # Reload is the same inference path used by CLI. Test predictions happen only after model freezing.
    result={};predictions={}
    for split in ['train','validation','test']:
        ii=idx[split];pred=predict(out/'model.npz',X[ii]);predictions[split]=pred
        result[split]={method:metric(y[ii],q) for method,q in pred.items()}
    r.dump(out/'metrics.json',result)
    with (out/'predictions.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.writer(f);w.writerow(['split','row','parent','live_time_s','method','target','truth_coefficient','prediction_coefficient'])
        for split,pred in predictions.items():
            for method,q in pred.items():
                for k,i in enumerate(idx[split]):
                    for j,t in enumerate(TARGETS):w.writerow([split,i,group[i],live[i],method,t,y[i,j],q[k,j]])
    strat={}
    for t in [60,600]:
        mask=live[te]==t;strat[f'live_{t}s']={m:metric(y[te][mask],p[mask]) for m,p in predictions['test'].items()}
    for j in range(3):
        for label,mask in [('low',y[te,j]<50),('high',y[te,j]>=50)]:
            strat[f'target_{j}_{label}']=dict(n=int(mask.sum()),metrics={m:metric(y[te][mask,j:j+1],p[mask,j:j+1]) for m,p in predictions['test'].items()})
    r.dump(out/'stratified_metrics.json',strat)
    fig,axes=plt.subplots(4,3,figsize=(11,12))
    for row,(method,q) in enumerate(predictions['test'].items()):
        for j in range(3):
            ax=axes[row,j];ax.scatter(y[te,j],q[:,j],s=6,alpha=.6);ax.plot([0,110],[0,110],'--',color='gray');ax.set(title=f'{method} | {TARGETS[j]}',xlabel='Synthetic truth',ylabel='Prediction')
    fig.suptitle('120 held-out mixtures x 2 noise realizations; fixed response ONLY');fig.tight_layout();fig.savefig(out/'test_scatter.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots();ax.semilogy(np.arange(1,501),history);ax.set(xlabel='Update index',ylabel='Training standardized MSE',title='Synthetic MLP fixed 500 updates');fig.tight_layout();fig.savefig(out/'training_loss.png',dpi=150);plt.close(fig)
    i=int(te[0]);pred={m:q[0] for m,q in predictions['test'].items()};fig,axes=plt.subplots(2,1,figsize=(10,7),sharex=True)
    axes[0].step(energy,X[i],label='Generated counts / live time',color='black')
    for m in ['nnls','mlp']:
        reconstruction=R@pred[m]+b;axes[0].plot(energy,reconstruction,label=m);axes[1].plot(energy,X[i]-reconstruction,label=m)
    axes[0].set(ylabel='Synthetic count rate',title=f'Test row {i}; same fixed response including known toy background');axes[0].legend();axes[1].axhline(0,color='gray');axes[1].set(xlabel='Energy [keV]',ylabel='Rate residual');axes[1].legend();fig.tight_layout();fig.savefig(out/'reconstruction.png',dpi=150);plt.close(fig)
    # Noiseless NNLS identity and serialized MLP equivalence are implementation checks, not benchmark scores.
    np.testing.assert_allclose(np.array([nnls(R,z-b)[0] for z in rate[:3]]),y[:3],atol=1e-9)
    np.testing.assert_allclose(predict(out/'model.npz',X[va])['mlp'],r.predict_mlp(p,s,X[va]))
    r.dump(out/'checks.json',dict(group_disjoint=True,unit_response_sums=R.sum(axis=0).tolist(),noiseless_nnls='PASS',reload_prediction='PASS',input_scaler_rows=len(tr),source_hash='PASS'))
    (out/'source').mkdir()
    for n in ['synthetic_response.py','audit_next_data.py','regression.py']:shutil.copy2(ROOT/n,out/'source'/n)
    print(json.dumps(result['test'],indent=2));infer(out,i)
def predict(path,X):
    with np.load(path,allow_pickle=False) as f:
        s={k:f[k] for k in ['x_mean','x_std','y_mean','y_std']};p={k:f[k] for k in ['W1','b1','W2','b2']}
        return dict(mean=np.tile(f['train_target_mean'],(len(X),1)),ridge=((X-s['x_mean'])/s['x_std'])@f['ridge_coef']*s['y_std']+s['y_mean'],mlp=r.predict_mlp(p,s,X),nnls=np.array([nnls(f['response'],x-f['background'])[0] for x in X]))
def infer(out,row):
    with np.load(out/'dataset.npz') as d:
        splits=json.loads((out/'splits.json').read_text());row=splits['test']['row_indices'][0] if row is None else row
        pred=predict(out/'model.npz',d['X_rate'][row:row+1]);result=dict(warning=WARNING,row=row,parent=int(d['parent_id'][row]),split=next(k for k,v in splits.items() if row in v['row_indices']),targets=TARGETS,truth=d['y_coefficients'][row].tolist(),predictions={k:v[0].tolist() for k,v in pred.items()})
    r.dump(out/'demo_prediction.json',result);print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('command',choices=['train','infer']);ap.add_argument('--out',type=Path,default=ROOT/'runs/synthetic-response-v1');ap.add_argument('--row',type=int);a=ap.parse_args()
    train(a.out) if a.command=='train' else infer(a.out,a.row)
