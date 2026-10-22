import glob, os, numpy as np, pandas as pd, streamlit as st
import matplotlib.pyplot as plt
from scipy import ndimage as ndi

DATA="demo_data"; st3=np.ones((3,3,3),int)
COL={1:(1,.5,0),2:(0,.8,0),3:(.85,0,0),4:(.6,.2,.8)}

@st.cache_data
def load(cid):
    d=np.load(f"{DATA}/{cid}.npz"); return d["t2"],d["gt"],d["pr"],d["sp"]

@st.cache_data
def lesions(cid):
    t2,gt,pr,sp=load(cid); vox=float(np.prod(sp))
    lab,n=ndi.label(gt==3,structure=st3)
    cav=ndi.distance_transform_edt(gt!=2,sampling=sp) if (gt==2).any() else None
    rows=[]
    for i in range(1,n+1):
        m=lab==i; v=m.sum()*vox
        rows.append({"id":i,"diam_mm":round((6*v/np.pi)**(1/3),1),
                     "dist_cavity_mm":round(float(cav[m].min()),1) if cav is not None else None,
                     "detected":bool((m&(pr==3)).any())})
    return lab,pd.DataFrame(rows)

def overlay(img,mask,a=.45):
    out=np.stack([img]*3,-1)/255.
    for k,c in COL.items():
        m=mask==k; out[m]=(1-a)*out[m]+a*np.array(c)
    return out

cases=sorted(os.path.basename(p)[:-4] for p in glob.glob(f"{DATA}/*.npz"))
st.title("FIRMA: fibroid map review (research prototype, not for clinical use)")
cid=st.sidebar.selectbox("Case",cases)
min_mm=st.sidebar.slider("Min lesion size (mm)",0,40,10)
t2,gt,pr,sp=load(cid); lab,df=lesions(cid)
df=df[df.diam_mm>=min_mm]; missed=df[~df.detected]
st.sidebar.metric("Lesions",len(df)); st.sidebar.metric("Missed by AI",len(missed))
pick=st.sidebar.selectbox("Jump to missed lesion",["-"]+missed.id.tolist())
k0=int(np.argmax((lab==pick).sum(axis=(0,1)))) if pick!="-" else t2.shape[2]//2
k=st.slider("Slice",0,t2.shape[2]-1,k0)
fig,axs=plt.subplots(1,3,figsize=(15,5))
for ax,(title,m) in zip(axs,[("T2",None),("Expert",gt),("AI model",pr)]):
    ax.imshow(np.rot90(overlay(t2[:,:,k],m[:,:,k]) if m is not None else t2[:,:,k]),cmap="gray")
    for i in missed.id:
        mk=lab[:,:,k]==i
        if mk.any(): ax.contour(np.rot90(mk).astype(float),levels=[.5],colors="yellow",linewidths=2)
    ax.set_title(title); ax.axis("off")
st.pyplot(fig)
st.caption("Yellow outline = expert-labelled lesion the AI missed. Orange=wall, green=cavity, red=myoma.")
st.dataframe(df.assign(status=np.where(df.detected,"found","MISSED")).drop(columns="detected"))