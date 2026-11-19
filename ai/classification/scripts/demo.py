"""Local CLI / Tkinter demo. No network, no uploads."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from prototype import ROOT,CLASSES,SmallCNN,cnn_scores,baseline_scores,getdata

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--run',default='runs/main'); ap.add_argument('--cli',action='store_true'); ap.add_argument('--smoke-gui',action='store_true'); ap.add_argument('--split',choices=['test_same','test_held'],default='test_held'); ap.add_argument('--index',type=int,default=0); args=ap.parse_args()
 run=ROOT/args.run; model=SmallCNN(); model.load_state_dict(torch.load(run/'models/cnn.pt',weights_only=True)); model.eval(); b=np.load(run/'models/baseline.npz'); datasets={s:getdata(run,s) for s in ['test_same','test_held']}
 def predict(split,index):
  d=datasets[split]
  if not 0<=index<len(d['labels']): raise ValueError('Sample index outside held-out dataset')
  x=d['counts'][index:index+1]; cs=cnn_scores(model,x)[0]; bs=baseline_scores(x,b['templates'],b['labels'],float(b['penalty']))[0]
  return dict(split=split,index=index,true=CLASSES[d['labels'][index]],cnn=CLASSES[cs.argmax()],baseline=CLASSES[bs.argmax()],cnn_scores=cs.tolist(),baseline_scores=bs.tolist(),total_counts=int(x.sum()),background_fraction=float(d['background_fraction'][index]),condition_index=int(d['condition_index'][index]))
 if args.cli:
  print(json.dumps(predict(args.split,args.index),ensure_ascii=False,indent=2)); print('Scores are not correctness probabilities. Five classes only; mixtures and unknown nuclides unsupported.'); return
 import tkinter as tk
 from tkinter import ttk,messagebox
 from matplotlib.figure import Figure
 from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
 root=tk.Tk(); root.title('GammaBench | 五类核素识别教学原型'); root.geometry('1120x820')
 ttk.Label(root,text='模拟单核素 + 实测本底形状 | 仅支持五类；不支持混合谱和库外核素的可靠识别',font=('Microsoft YaHei',12)).pack(pady=10)
 ttk.Label(root,text='CNN 分数为未校准 Softmax，模板分数为平均对数似然；均不代表“判断正确的概率”。').pack()
 panel=ttk.Frame(root); panel.pack(pady=8); split=tk.StringVar(value=args.split); idx=tk.IntVar(value=args.index)
 ttk.Combobox(panel,textvariable=split,values=list(datasets),state='readonly',width=18).pack(side='left'); ttk.Label(panel,text=' 测试样本编号 ').pack(side='left'); spin=ttk.Spinbox(panel,from_=0,to=len(datasets[args.split]['labels'])-1,textvariable=idx,width=10); spin.pack(side='left')
 info=tk.StringVar(); ttk.Label(root,textvariable=info,font=('Microsoft YaHei',11)).pack()
 fig=Figure(figsize=(10,5),layout='constrained'); ax=fig.add_subplot(111); canvas=FigureCanvasTkAgg(fig,master=root); canvas.get_tk_widget().pack(fill='both',expand=True)
 tree=ttk.Treeview(root,columns=('class','cnn','template'),show='headings',height=5)
 for key,title in [('class','类别'),('cnn','CNN 未校准评分'),('template','模板平均对数似然评分')]: tree.heading(key,text=title); tree.column(key,width=240,anchor='center')
 tree.pack(pady=8)
 def refresh():
  try:
   s=split.get(); i=idx.get(); z=predict(s,i); d=datasets[s]
   info.set(f'真实：{z["true"]}    CNN：{z["cnn"]}    模板：{z["baseline"]}\n总计数 {z["total_counts"]} | 期望本底份额 {z["background_fraction"]:.1%} | 响应索引 {z["condition_index"]}')
   ax.clear(); ax.step(20+2*np.arange(1024),d['counts'][i],where='post',lw=.8); ax.set(xlabel='Energy (keV; repository convention)',ylabel='Observed counts / 2 keV bin',title=f'{s} / sample {i}'); ax.grid(alpha=.2); canvas.draw()
   for item in tree.get_children(): tree.delete(item)
   for k,c in enumerate(CLASSES): tree.insert('','end',values=(c,f'{z["cnn_scores"][k]:.5f}',f'{z["baseline_scores"][k]:.5f}'))
  except Exception as e: messagebox.showerror('输入错误',str(e))
 def next_error():
  s=split.get(); d=datasets[s]; preds=np.load(run/'data'/f'{s}_predictions.npz')['cnn'].argmax(1); bad=np.flatnonzero(preds!=d['labels']); after=bad[bad>idx.get()]
  if len(bad): idx.set(int(after[0] if len(after) else bad[0])); refresh()
 ttk.Button(panel,text='显示 / 预测',command=refresh).pack(side='left',padx=5); ttk.Button(panel,text='下一个 CNN 误判',command=next_error).pack(side='left'); refresh()
 if args.smoke_gui:
  root.update(); fig.savefig(run/'figures/demo_preview.png',dpi=130); root.destroy(); print('GUI controls, canvas and inference smoke test passed')
 else: root.mainloop()
if __name__=='__main__': main()
