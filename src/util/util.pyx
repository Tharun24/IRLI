import numpy as np
cimport numpy as np
import cython

cdef extern from "Gather.h":
    void cgather_batch(float*, long*, float*, long*, int, int, int, int, int) except +
    void cgather_K(float*, long*, float*, long*, int, int, int, int,int) except +
    void ctopk(float*, long*, int, int, int,int) except +
    void ctopk_csr(long*, long*, float*, long*, int, int,int) except +

@cython.boundscheck(False)
def gather_batch(np.ndarray[float, ndim=3, mode="c"] raw,
           np.ndarray[long, ndim=2, mode="c"] indices,
           np.ndarray[float, ndim=2, mode="c"] scores,
           np.ndarray[long, ndim=2, mode="c"] top_preds,
           int R, int B, int N, int batch_size, int n_threads):
    cgather_batch(&raw[0,0,0], &indices[0,0], &scores[0,0], &top_preds[0,0], R, B, N, batch_size, n_threads)

@cython.boundscheck(False)
def gather_K(np.ndarray[float, ndim=3, mode="c"] raw,
           np.ndarray[long, ndim=2, mode="c"] indices,
           np.ndarray[float, ndim=2, mode="c"] scores,
           np.ndarray[long, ndim=2, mode="c"] top_preds,
           int R, int B, int N, int batch_size, int n_threads):
    cgather_K(&raw[0,0,0], &indices[0,0], &scores[0,0], &top_preds[0,0], R, B, N, batch_size, n_threads)

@cython.boundscheck(False)
def topK(np.ndarray[float, ndim=2, mode="c"] scores,
           np.ndarray[long, ndim=2, mode="c"] top_preds,
           int B, int N, int K, int n_threads):
    ctopk(&scores[0,0], &top_preds[0,0], B, N, K, n_threads)

@cython.boundscheck(False)
def topk_csr(
           np.ndarray[long, ndim=1] indices,
           np.ndarray[long, ndim=1] indptr,
           np.ndarray[float, ndim=1] data,
           np.ndarray[long, ndim=2, mode="c"] top_preds,
           int batch_size, int topk, int n_threads):
    ctopk_csr(&indices[0], &indptr[0], &data[0], &top_preds[0,0], batch_size, topk, n_threads)

