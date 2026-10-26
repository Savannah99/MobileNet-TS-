"""Real image inference: shared tensors for Chole/Auto, staged sources for Pedi."""
import time
from concurrent.futures import ThreadPoolExecutor
from base import *
from efficient_inference import Confidence, BatchPreprocessor, read_uint8_batch


class RichConfidence(Confidence):
    def _forward(self,x):
        s=self.scorer
        h=s.trunk(s.pool(s.features(x)).flatten(1))
        p=s.cls_head(h).softmax(1).max(1).values
        s.flow_head(h)
        return torch.cat([p[:,None],h],1)


def select_joint(confidence,features,k,beta):
    assert k in (4,8)
    if beta==0:
        return indices("tsplus",len(confidence),k,confidence)
    n=len(confidence)//16;b=16//k
    p=np.asarray(confidence[:n*16],dtype=np.float64).reshape(n,16)
    h=np.array(features[:n*16],dtype=np.float64,order="C",copy=True)
    h/=np.maximum(np.linalg.norm(h,axis=1,keepdims=True),1e-12)
    h=h.reshape(n,16,64)
    chosen=np.empty((n,k),dtype=np.int64);rows=np.arange(n)
    chosen[:,0]=p[:,:b].argmax(1)
    for i in range(1,k):
        last=h[rows,chosen[:,i-1]]
        novelty=np.clip(1-(h[:,i*b:(i+1)*b]*last[:,None]).sum(-1),0,1)
        chosen[:,i]=i*b+(p[:,i*b:(i+1)*b]+beta*novelty).argmax(1)
    return chosen+np.arange(n)[:,None]*16


@torch.inference_mode()
def extract(ds,vid,scorer,pool):
    prep=BatchPreprocessor();parts=[];n=len(ds.labels(vid))
    for start in range(0,n,128):
        cpu=read_uint8_batch([ds.image(vid,i) for i in range(start,min(start+128,n))],pool)
        parts.append(scorer(prep(cpu)).cpu().numpy())
    return np.concatenate(parts)


@torch.inference_mode()
def process(ds,vid,k,beta,encoder,scoring,model,pool,collect=False):
    prep=BatchPreprocessor();loader=ThreadPoolExecutor(max_workers=1)
    n=len(ds.labels(vid));cut=n//16*16
    def stream(paths):
        fut=loader.submit(read_uint8_batch,paths[0],pool)
        for i in range(len(paths)):
            cpu=fut.result()
            fut=loader.submit(read_uint8_batch,paths[i+1],pool) if i+1<len(paths) else None
            yield cpu
    def choose(signals):
        return select_joint(signals if signals.ndim==1 else signals[:,0],
                            None if signals.ndim==1 else signals[:,1:],k,beta)
    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
    started=time.perf_counter();features=[];index_parts=[];signal_parts=[]
    try:
        if ds.name=="pedi38" and k!=16:
            paths=[[ds.image(vid,i) for i in range(s,min(s+128,n))] for s in range(0,n,128)]
            signals=np.concatenate([scoring(prep(cpu)).cpu().numpy() for cpu in stream(paths)])
            ix=choose(signals)
            paths=[[encoder_image(ds,vid,int(i)) for i in ix[s:s+8].flat] for s in range(0,len(ix),8)]
            for cpu in stream(paths):
                features.append(encoder(prep(cpu)).reshape(-1,k,384).mean(1))
            if collect:signal_parts.append(signals)
        else:
            length=cut if k==16 else n
            paths=[[encoder_image(ds,vid,i) for i in range(s,min(s+128,length))]
                   for s in range(0,length,128)]
            for chunk,cpu in enumerate(stream(paths)):
                x=prep(cpu)
                if k==16:
                    local=indices("dense",len(x),16)
                else:
                    signals=scoring(x).cpu().numpy();local=choose(signals)
                    if collect:signal_parts.append(signals)
                if len(local):
                    if k==16:
                        # Dense already owns precisely the complete-window tensor.
                        encoded=encoder(x)
                    else:
                        selected=torch.from_numpy(local.reshape(-1)).cuda()
                        encoded=encoder(x.index_select(0,selected))
                    features.append(encoded.reshape(-1,k,384).mean(1))
                    index_parts.append(local+chunk*128)
                del x
            ix=np.concatenate(index_parts)
        f=torch.cat(features)[None]
        pred=model(f).argmax(1)[0].cpu().numpy()
        torch.cuda.synchronize();elapsed=time.perf_counter()-started
        result={"seconds":elapsed,"windows":len(ix),"ms_per_window":elapsed/len(ix)*1000,
                "peak_allocated_mib":torch.cuda.max_memory_allocated()/2**20,
                "predictions":pred,"selected_indices":ix,"encoded_frames":int(ix.size),
                "scored_frames":n if k!=16 else 0}
        if collect:
            result["pooled_features"]=f[0].cpu().numpy()
            if signal_parts:result["signals"]=np.concatenate(signal_parts)
        return result
    finally:
        loader.shutdown(wait=True,cancel_futures=True)
