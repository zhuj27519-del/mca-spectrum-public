"""LaBr3 eight-sample empirical regression v1. NumPy CPU, no physical calibration."""
from pathlib import Path
import argparse,csv,hashlib,json,platform
import numpy as np
from openpyxl import load_workbook
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
VERSION='labr3-regression-v1'
WARNING='8样品探索性模型，未经独立实测验证。'
TARGETS=['Ra-226','Th-228','K-40']
EXPECTED_SHA='4d5d34837bc698b29e853d7da6d82e40fbd21d3fc0b1cbd028f6497eb8fefb27'
CONFIG=dict(version=VERSION,seed=20261116,epochs=500,learning_rate=.001,
    architecture=[64,8,3],hidden='ReLU',output='Softplus',optimizer='full-batch Adam',
    adam_beta1=.9,adam_beta2=.999,adam_epsilon=1e-8,weight_decay=0,
    input_scaling='training-fold mean and population std, zero std -> 1',
    target_scaling='training-fold mean and population std; loss compares standardized predictions and targets',
    positive_output='prediction_Bq_kg = target_std * Softplus(logits); standardized prediction = Softplus(logits) - target_mean/target_std',
    ridge_alpha=1.0,ridge_objective='sum squared standardized errors + alpha * sum squared coefficients; intercept unpenalized',
    split='leave-one-named-sample-out, 8 folds; source groups unconfirmed',
    aggregation='sum consecutive blocks of 16 channels after division by live time: 1024 -> 64',
    background_subtraction=False,augmentation=False,early_stopping=False,
    hyperparameter_selection='fixed before first evaluation; no held-out tuning',
    targets=TARGETS,units='Bq/kg',source_sha256=EXPECTED_SHA,warning=WARNING)

def dump(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')

def load_samples(path):
    assert hashlib.sha256(path.read_bytes()).hexdigest()==EXPECTED_SHA,'Unreviewed workbook hash'
    w=load_workbook(path,data_only=True,keep_links=False)
    m,s=w['meta-data'],w['Spectra']
    columns={s.cell(1,j).value:j for j in range(3,15)}
    names=[];counts=[];times=[];targets=[];uncertainties=[];masses=[]
    for r in range(6,14):
        name=m.cell(r,1).value;j=columns[name]
        names.append(name);counts.append([s.cell(i,j).value for i in range(2,1026)])
        times.append(m.cell(r,2).value);masses.append(m.cell(r,3).value)
        pairs=[[float(v.strip()) for v in m.cell(r,k).value.split('±')] for k in [6,7,8]]
        targets.append([p[0] for p in pairs]);uncertainties.append([p[1] for p in pairs])
    counts=np.asarray(counts,dtype=float);times=np.asarray(times,dtype=float)
    assert counts.shape==(8,1024) and len(set(names))==8
    assert np.isfinite(counts).all() and (counts>=0).all() and (times>0).all()
    X=features(counts,times)
    energy=np.array([s.cell(i,2).value for i in range(2,1026)],float)
    return names,X,np.array(targets),np.array(uncertainties),np.array(masses),energy

def features(counts,live):
    counts=np.asarray(counts,dtype=float);live=np.asarray(live,dtype=float)
    if counts.ndim!=2 or counts.shape[1]!=1024:raise ValueError('Expected n x 1024 counts')
    if not np.isfinite(counts).all() or (counts<0).any():raise ValueError('Expected finite nonnegative raw counts; no clipping applied')
    if live.shape!=(len(counts),) or not np.isfinite(live).all() or (live<=0).any():raise ValueError('Expected positive live times')
    rates=counts/live[:,None];X=rates.reshape(len(counts),64,16).sum(axis=2)
    np.testing.assert_allclose(X.sum(axis=1),rates.sum(axis=1),rtol=1e-13)
    return X

def scaling(X,y):
    xs=X.std(axis=0);ys=y.std(axis=0)
    return dict(x_mean=X.mean(axis=0),x_std=np.where(xs==0,1,xs),y_mean=y.mean(axis=0),y_std=np.where(ys==0,1,ys))

def initialize(seed):
    rng=np.random.default_rng(seed)
    return dict(W1=rng.normal(size=(64,8))*np.sqrt(2/64),b1=np.zeros(8),
                W2=rng.normal(size=(8,3))*np.sqrt(1/8),b2=np.zeros(3))

def forward(p,X):
    hidden=np.maximum(X@p['W1']+p['b1'],0)
    return np.logaddexp(0,hidden@p['W2']+p['b2'])

def loss_gradient(p,X,target_z,target_offset):
    a=X@p['W1']+p['b1'];h=np.maximum(a,0);z=h@p['W2']+p['b2']
    output=np.logaddexp(0,z)
    err=(output-target_offset)-target_z
    loss=np.mean(err**2)
    dz=2*err/err.size*np.exp(-np.logaddexp(0,-z))
    dh=(dz@p['W2'].T)*(a>0)
    return float(loss),dict(W1=X.T@dh,b1=dh.sum(axis=0),W2=h.T@dz,b2=dz.sum(axis=0))

def fit_mlp(X,y,scale):
    # This function accepts training rows only. No held-out target or validation loss.
    Xz=(X-scale['x_mean'])/scale['x_std']
    yz=(y-scale['y_mean'])/scale['y_std'];offset=scale['y_mean']/scale['y_std']
    p=initialize(CONFIG['seed']);m={k:np.zeros_like(v) for k,v in p.items()};v={k:np.zeros_like(q) for k,q in p.items()}
    history=[]
    for step in range(1,CONFIG['epochs']+1):
        loss,g=loss_gradient(p,Xz,yz,offset);history.append(loss)
        for k in p:
            m[k]=.9*m[k]+.1*g[k];v[k]=.999*v[k]+.001*g[k]**2
            p[k]-=CONFIG['learning_rate']*(m[k]/(1-.9**step))/(np.sqrt(v[k]/(1-.999**step))+1e-8)
    assert all(np.isfinite(q).all() for q in p.values())
    return p,history

def predict_mlp(p,scale,X):
    return forward(p,(X-scale['x_mean'])/scale['x_std'])*scale['y_std']

def fit_ridge(X,y,scale):
    Z=(X-scale['x_mean'])/scale['x_std'];T=(y-scale['y_mean'])/scale['y_std']
    coef=np.linalg.solve(Z.T@Z+CONFIG['ridge_alpha']*np.eye(64),Z.T@T)
    return coef

def save_model(path,p,scale):
    np.savez_compressed(path,**p,**scale,version=np.array(VERSION),targets=np.array(TARGETS),warning=np.array(WARNING))

def train(source,out):
    if out.exists():raise FileExistsError('Use a fresh --out directory to preserve completed runs')
    out.mkdir(parents=True);(out/'folds').mkdir()
    dump(out/'config.json',CONFIG) # Freeze config before any evaluation.
    names,X,y,uncertainty,mass,energy=load_samples(source)
    np.savez_compressed(out/'dataset.npz',names=names,X_cps=X,y_Bq_kg=y,label_reported_plusminus=uncertainty,mass_g=mass)
    intervals=[dict(bin=i,channel_first=16*i,channel_last=16*i+15,first_listed_energy_keV=float(energy[16*i]),last_listed_energy_keV=float(energy[16*i+15])) for i in range(64)]
    dump(out/'energy_intervals.json',intervals)
    predictions={k:np.empty_like(y) for k in ['mean','ridge','mlp']};splits=[]
    for test in range(8):
        tr=np.array([i for i in range(8) if i!=test]);scale=scaling(X[tr],y[tr])
        p,hist=fit_mlp(X[tr],y[tr],scale);coef=fit_ridge(X[tr],y[tr],scale)
        predictions['mean'][test]=y[tr].mean(axis=0)
        predictions['ridge'][test]=((X[test]-scale['x_mean'])/scale['x_std'])@coef*scale['y_std']+scale['y_mean']
        predictions['mlp'][test]=predict_mlp(p,scale,X[test:test+1])[0]
        np.savez_compressed(out/'folds'/f'fold_{test}.npz',**p,**scale,ridge_coef=coef,train_indices=tr,test_index=test,training_loss_before_update=hist)
        splits.append(dict(fold=test,train_names=[names[i] for i in tr],test_name=names[test],train_indices=tr.tolist(),test_index=test))
    dump(out/'splits.json',splits)
    with (out/'oof_predictions.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.writer(f);w.writerow(['sample','fold','method','target','truth_Bq_kg','prediction_Bq_kg','error_prediction_minus_truth_Bq_kg'])
        for method,pred in predictions.items():
            for i,name in enumerate(names):
                for j,target in enumerate(TARGETS):w.writerow([name,i,method,target,y[i,j],pred[i,j],pred[i,j]-y[i,j]])
    metrics=[]
    for method,pred in predictions.items():
        error=pred-y
        for j,target in enumerate(TARGETS):
            metrics.append(dict(method=method,target=target,n=8,MAE=float(np.mean(np.abs(error[:,j]))),RMSE=float(np.sqrt(np.mean(error[:,j]**2))),bias=float(error[:,j].mean())))
    dump(out/'metrics.json',metrics)
    with (out/'metrics.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(metrics[0]));w.writeheader();w.writerows(metrics)
    fig,axes=plt.subplots(3,3,figsize=(12,11))
    for row,(method,pred) in enumerate(predictions.items()):
        for j,target in enumerate(TARGETS):
            ax=axes[row,j];lo=min(0,y[:,j].min(),pred[:,j].min());hi=max(y[:,j].max(),pred[:,j].max())*1.08
            ax.plot([lo,hi],[lo,hi],'--',color='gray');ax.scatter(y[:,j],pred[:,j])
            for i in range(8):ax.annotate(str(i+1),(y[i,j],pred[i,j]),fontsize=8,xytext=(3,3),textcoords='offset points')
            ax.set(xlabel='HPGe reference [Bq/kg]',ylabel='OOF prediction [Bq/kg]',title=f'{method} | {target}',xlim=(lo,hi),ylim=(lo,hi));ax.grid(alpha=.2)
    fig.suptitle('Eight-sample leave-one-out evaluation; no independent external validation')
    fig.tight_layout(rect=(0,.06,1,.97));fig.text(.05,.01,'Sample IDs: '+', '.join(f'{i+1}={n}' for i,n in enumerate(names)),fontsize=8,wrap=True)
    fig.savefig(out/'oof_scatter.png',dpi=150);plt.close(fig)
    # Only now fit demonstration model on ALL rows. It never enters OOF metrics.
    full_scale=scaling(X,y);full_p,history=fit_mlp(X,y,full_scale)
    save_model(out/'demo_mlp.npz',full_p,full_scale)
    dump(out/'demo_training_history.json',{'kind':'training loss only, not generalization', 'loss_before_update':history})
    dump(out/'environment.json',{'python':platform.python_version(),'numpy':np.__version__,'matplotlib':matplotlib.__version__,'platform':platform.platform(),'execution':'CPU float64 full-batch NumPy'})
    print(json.dumps(metrics,indent=2));print(WARNING)

def infer(source,model,sample):
    names,X,y,_,_,_=load_samples(source)
    if sample not in names:raise ValueError('Choose one of '+', '.join(names))
    i=names.index(sample)
    with np.load(model,allow_pickle=False) as f:
        assert str(f['version'])==VERSION
        p={k:f[k] for k in ['W1','b1','W2','b2']};scale={k:f[k] for k in ['x_mean','x_std','y_mean','y_std']}
    prediction=predict_mlp(p,scale,X[i:i+1])[0]
    result=dict(warning=WARNING,model=str(model),sample=sample,kind='ALL-8 model demonstration on a training sample; NOT OOF evaluation',
                targets=TARGETS,units='Bq/kg',prediction=prediction.tolist(),HPGe_reference=y[i].tolist(),
                applicability='Same LaBr3 setup only; geometry/background/attenuation/source grouping limitations remain')
    print(json.dumps(result,ensure_ascii=False,indent=2));return result

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=['train','infer'])
    parser.add_argument('--source',type=Path,default=ROOT/'data/raw/labr3-NORM-samples-spectra.xlsx')
    parser.add_argument('--out',type=Path,default=ROOT/'runs'/VERSION)
    parser.add_argument('--model',type=Path,default=ROOT/'runs'/VERSION/'demo_mlp.npz')
    parser.add_argument('--sample',default='Fly-ash-1');parser.add_argument('--save-json',type=Path)
    a=parser.parse_args()
    if a.command=='train':train(a.source,a.out)
    else:
        result=infer(a.source,a.model,a.sample)
        if a.save_json:dump(a.save_json,result)
