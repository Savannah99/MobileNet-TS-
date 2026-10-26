"""FP32 JPEG inference with batched preprocessing and reusable Cholec inputs.

Keeps the original PIL resize/crop, image sources, scoring batches and models.
No score caches are accepted for a real scorer. An explicitly marked Cholec
timing proxy may replay historical selections *after* charging live scoring.
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from PIL import Image

from common import DEVICE, WINDOW
from vision import TRANSFORM, encoder_image


GEOMETRY = TRANSFORM.transforms[:2]


def read_uint8(path):
    """Exactly the reference's PIL RGB/resize/center-crop, without per-image torch ops."""
    with Image.open(path) as source:
        image = source.convert("RGB")
        for transform in GEOMETRY:
            image = transform(image)
        return np.array(image, dtype=np.uint8)


def read_uint8_batch(paths, pool):
    # NHWC contiguous bytes: 1/4 of the original FP32 H2D payload.
    return torch.from_numpy(np.stack(list(pool.map(read_uint8, paths))))


class BatchPreprocessor:
    def __init__(self):
        # GPU arithmetic can differ by <1e-6 from the CPU reference. In TF32
        # convolutions that is enough to change nearly tied confidence ranks.
        # RGB JPEG values are bytes, so a 3x256 table reproduces *exactly* the
        # original CPU FP32 ToTensor/Normalize outputs without per-image work.
        table = torch.arange(256, dtype=torch.float32).repeat(3,1).div_(255)
        table.sub_(torch.tensor([.485,.456,.406])[:,None])
        table.div_(torch.tensor([.229,.224,.225])[:,None])
        self.table = table.flatten().to(DEVICE)
        self.channel_offset = (torch.arange(3,device=DEVICE,dtype=torch.int32)*256).view(1,3,1,1)

    def __call__(self, cpu):
        x = cpu.to(DEVICE, non_blocking=cpu.is_pinned())
        x = x.permute(0,3,1,2).contiguous().to(torch.int32)
        x.add_(self.channel_offset)
        # index_select accepts int32 directly. Advanced tensor indexing can
        # allocate a full int64 index copy, needlessly increasing peak memory.
        return self.table.index_select(0,x.reshape(-1)).view(x.shape)


class Confidence:
    """Optional CUDA replay for the unchanged FP32 64-frame scorer.

    Fixed-size output is consumed or cloned before the next replay. Tails use
    their original batch size and eager execution; no padding changes BN/conv
    shapes. Setup/capture is deliberately separate from steady-state timing.
    """
    def __init__(self, scorer, cuda_graph=False):
        self.scorer = scorer
        self.graph = None
        self.setup_seconds = 0.
        if cuda_graph:
            started = time.perf_counter()
            self.static_input = torch.zeros(64, 3, 224, 224, device=DEVICE)
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.inference_mode(), torch.cuda.stream(stream):
                for _ in range(3):
                    self._forward(self.static_input)
            torch.cuda.current_stream().wait_stream(stream)
            torch.cuda.synchronize()
            self.graph = torch.cuda.CUDAGraph()
            with torch.inference_mode(), torch.cuda.graph(self.graph):
                self.static_output = self._forward(self.static_input)
            torch.cuda.synchronize()
            self.setup_seconds = time.perf_counter() - started

    def _forward(self, x):
        return self.scorer(x)[0].softmax(1).max(1).values

    def __call__(self, x):
        parts = []
        for part in x.split(64):
            if self.graph is not None and len(part) == 64:
                self.static_input.copy_(part)
                self.graph.replay()
                parts.append(self.static_output.clone())
            else:
                parts.append(self._forward(part))
        return torch.cat(parts)


@torch.inference_mode()
def process_video(ds, vid, method, encoder, confidence, gru, pool,
                  *, chunk_windows=8, selection_signal_override=None,
                  collect=False, profile=False, prefetch=False, staged_pedi=False):
    """End-to-end JPEG processing; timestamps and all JPEG loads remain in scope.

    A single prefetch batch, when enabled, overlaps CPU image geometry with
    current GPU work. Dense/fixed and TS+ use the same reader/prefetch policy.
    Profile mode synchronizes stage boundaries for diagnostics only.
    """
    if method not in ("dense", "fixed", "tsplus"):
        raise ValueError(method)
    if staged_pedi and ds.name=="pedi38" and method=="tsplus":
        if selection_signal_override is not None:
            raise ValueError("Pedi staged inference never accepts cached scoring")
        return process_pedi_staged(ds,vid,encoder,confidence,gru,pool,
                                  chunk_windows=chunk_windows,collect=collect,
                                  prefetch=prefetch,profile=profile)
    if selection_signal_override is not None:
        if method != "tsplus" or not confidence.scorer.timing_proxy_head:
            raise ValueError("Replay is only allowed for the explicit missing-head timing proxy")
    n = len(ds.labels(vid)) // WINDOW * WINDOW
    preprocessor = BatchPreprocessor()
    if selection_signal_override is not None and len(selection_signal_override) < n:
        raise ValueError("Incomplete historical trace")
    parts = dict(candidate_jpeg=0., candidate_h2d_normalize=0., scorer_select=0.,
                 encoder_jpeg=0., encoder_h2d_normalize=0., vit_pool=0., bigru=0.)
    selected, scores, pooled = [], [], []
    chunks = [(s, min(s+WINDOW*chunk_windows, n)) for s in range(0,n,WINDOW*chunk_windows)]

    def first_stage(bounds):
        start, end = bounds
        if method == "tsplus":
            paths = [ds.image(vid,i) for i in range(start,end)]
        else:
            offset = range(WINDOW) if method == "dense" else range(1,WINDOW,2)
            paths = [encoder_image(ds,vid,i+j) for i in range(start,end,WINDOW) for j in offset]
        return read_uint8_batch(paths,pool)

    def finish_stage(key, t):
        if profile:
            torch.cuda.synchronize()
        parts[key] += time.perf_counter()-t

    loader = ThreadPoolExecutor(max_workers=1) if prefetch else None
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    try:
        future = loader.submit(first_stage,chunks[0]) if loader and chunks else None
        for ci,(start,end) in enumerate(chunks):
            t = time.perf_counter()
            cpu = future.result() if future else first_stage((start,end))
            if loader and ci+1 < len(chunks):
                future = loader.submit(first_stage,chunks[ci+1])
            else:
                future = None
            finish_stage("candidate_jpeg",t)
            t = time.perf_counter()
            x = preprocessor(cpu)
            finish_stage("candidate_h2d_normalize",t)
            if method == "tsplus":
                t = time.perf_counter()
                signal = confidence(x)
                # Keep selection on GPU when images can be reused.
                choose_signal = signal if selection_signal_override is None else torch.as_tensor(
                    selection_signal_override[start:end], device=DEVICE)
                local_idx = choose_signal.reshape(-1,2).argmax(1) + torch.arange(
                    0,end-start,2,device=DEVICE)
                if collect:
                    scores.append(signal.cpu().numpy())
                    selected.append(local_idx.cpu().numpy().reshape(-1,8)+start)
                finish_stage("scorer_select",t)
                if ds.name == "cholec80":
                    imgs = x.index_select(0,local_idx)
                    del x
                else:
                    # The scorer input is dead. Release it before allocating
                    # the distinct full-resolution encoder batch.
                    del x
                    # Pedi scorer thumbnails are NOT encoder images: retain
                    # the full-resolution JPEG source even though it's slower.
                    t = time.perf_counter()
                    host_idx = local_idx.cpu().numpy()
                    encoded_cpu = read_uint8_batch(
                        [encoder_image(ds,vid,start+int(i)) for i in host_idx],pool)
                    finish_stage("encoder_jpeg",t)
                    t = time.perf_counter()
                    imgs = preprocessor(encoded_cpu)
                    finish_stage("encoder_h2d_normalize",t)
                    del encoded_cpu
                slots = 8
                del signal,choose_signal,local_idx
            else:
                imgs = x
                del x
                slots = 16 if method == "dense" else 8
                if collect:
                    offsets = np.arange(16) if method == "dense" else np.arange(1,16,2)
                    selected.append(np.arange(start,end,16)[:,None]+offsets[None,:])
            t = time.perf_counter()
            f = encoder(imgs).reshape(-1,slots,384).mean(1)
            # Keep pooled features on GPU until the complete BiGRU sequence.
            pooled.append(f)
            finish_stage("vit_pool",t)
            del cpu,imgs,f
        t = time.perf_counter()
        features = torch.cat(pooled).unsqueeze(0)
        pred = gru(features).argmax(1)[0].cpu().numpy()
        finish_stage("bigru",t)
        torch.cuda.synchronize()
        elapsed = time.perf_counter()-started
        row = dict(seconds=elapsed,windows=n//16,ms_per_window=elapsed/(n//16)*1000,
                   peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                   predictions=pred,profile=profile,prefetch=prefetch)
        if profile:
            row["parts_seconds"] = parts
        if collect:
            row.update(selected_indices=np.concatenate(selected),
                       scores=np.concatenate(scores) if scores else None,
                       pooled_features=features[0].cpu().numpy())
        return row
    finally:
        if loader:
            loader.shutdown(wait=True,cancel_futures=True)


@torch.inference_mode()
def process_pedi_staged(ds,vid,encoder,confidence,gru,pool,
                        *,chunk_windows=8,collect=False,prefetch=True,profile=False):
    """Offline two-pass schedule for Pedi's distinct thumbnail/original sources.

    First score the thumbnails (unchanged 64-frame neural batches), then stream
    the selected original JPEGs with CPU prefetch. This avoids blocking every
    encoder batch on the next thumbnail/scoring round. Both passes are timed;
    only small index arrays, not images, are retained between passes.
    """
    if ds.name!="pedi38":
        raise ValueError("The staged schedule is for Pedi's two image sources")
    n=len(ds.labels(vid))//16*16
    bounds=[(s,min(s+16*chunk_windows,n)) for s in range(0,n,16*chunk_windows)]
    prep=BatchPreprocessor()
    loader=ThreadPoolExecutor(max_workers=1) if prefetch else None
    selected,scores,pooled=[],[],[]

    def stream_batches(path_batches):
        def load(paths):
            return read_uint8_batch(paths,pool)
        future=loader.submit(load,path_batches[0]) if loader else None
        for i,paths in enumerate(path_batches):
            cpu=future.result() if future else load(paths)
            future=loader.submit(load,path_batches[i+1]) if loader and i+1<len(path_batches) else None
            yield cpu

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started=time.perf_counter()
    try:
        thumbnail_paths=[[ds.image(vid,i) for i in range(s,e)] for s,e in bounds]
        for (start,end),cpu in zip(bounds,stream_batches(thumbnail_paths)):
            x=prep(cpu)
            signal=confidence(x)
            ix=signal.reshape(-1,2).argmax(1)+torch.arange(0,end-start,2,device=DEVICE)
            selected.append(ix.cpu().numpy().reshape(-1,8)+start)
            if collect:
                scores.append(signal.cpu().numpy())
            del cpu,x,signal,ix
        scoring_seconds=time.perf_counter()-started
        encode_started=time.perf_counter()
        paths=[[encoder_image(ds,vid,int(i)) for i in ix.flat] for ix in selected]
        for cpu in stream_batches(paths):
            imgs=prep(cpu)
            pooled.append(encoder(imgs).reshape(-1,8,384).mean(1))
            del cpu,imgs
        if profile:
            torch.cuda.synchronize()
        encoding_seconds=time.perf_counter()-encode_started
        t=time.perf_counter()
        features=torch.cat(pooled).unsqueeze(0)
        pred=gru(features).argmax(1)[0].cpu().numpy()
        torch.cuda.synchronize()
        elapsed=time.perf_counter()-started
        row=dict(seconds=elapsed,windows=n//16,ms_per_window=elapsed/(n//16)*1000,
                 peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                 predictions=pred,profile=profile,prefetch=prefetch,staged_pedi=True)
        if profile:
            row["parts_seconds"]=dict(thumbnail_scoring_pass=scoring_seconds,
                                       selected_encoding_pass=encoding_seconds,
                                       bigru=time.perf_counter()-t)
        if collect:
            row.update(selected_indices=np.concatenate(selected),scores=np.concatenate(scores),
                       pooled_features=features[0].cpu().numpy())
        return row
    finally:
        if loader:
            loader.shutdown(wait=True,cancel_futures=True)
