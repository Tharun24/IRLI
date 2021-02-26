#include <queue>
#include "Gather.h"
#include <iostream>
using namespace std;
// raw - M, R, B
// indices - R, N
// result = M, N
void cgather_batch(float* raw, long* lookup, float* result, long* top_preds, int R, int B, int N, int batch_size, int n_threads)
{
    vector<priority_queue<pair<float, long>>> q(batch_size);
    #pragma omp parallel for num_threads(n_threads)   
    for(int idx = 0; idx < batch_size; ++idx)
    {
        const int preds_offset = idx * 5;
        const int scores_offset = idx * N;    
        for(int rdx = 0; rdx < R; ++rdx)
        {
            const int idx_offset = rdx * N;
            const int raw_offset = idx * R * B + rdx * B;

            for(int kdx = 0; kdx < N; ++kdx)
            {
                result[scores_offset + kdx] += raw[raw_offset + lookup[idx_offset + kdx]];
            }
        }
        // filling the queue
        for(int i = 0; i < N; ++i)
        {
            if(q[idx].size()<100)
                q[idx].push(pair<float, long>(-result[scores_offset + i], i));
            else if(q[idx].top().first > -result[scores_offset + i]){
                q[idx].pop();
                q[idx].push(pair<float, long>(-result[scores_offset + i], i));
            }    
        }
        // getting the top 100 classes
        for(long i = 99; i >=0 ; --i)
        {
            top_preds[preds_offset + i] = q[idx].top().second;
            q[idx].pop();
        }
    }
}

void cgather_K(float* raw, long* lookup, float* result, long* top_preds, int R, int B, int N, int batch_size, int n_threads)
{       
    vector<priority_queue<pair<float, long>>> q(batch_size);  
    for(int idx = 0; idx < batch_size; ++idx)
    {
        const int preds_offset = idx * 5;
        const int scores_offset = idx * N;    
        for(int rdx = 0; rdx < R; ++rdx)
        {
            const int idx_offset = rdx * N;
            const int raw_offset = idx * R * B + rdx * B;

            #pragma omp parallel for num_threads(n_threads) 
            for(int kdx = 0; kdx < N; ++kdx)
            {
                result[scores_offset + kdx] += raw[raw_offset + lookup[idx_offset + kdx]];
            }
        }
        // filling the queue
        for(long i = 0; i < N; ++i)
        {
            if(q[idx].size()<5)
                q[idx].push(pair<float, long>(-result[scores_offset + i], i));
            else if(q[idx].top().first > -result[scores_offset + i]){
                q[idx].pop();
                q[idx].push(pair<float, long>(-result[scores_offset + i], i));
            }    
        }
        // getting the top 5 classes
        for(long i = 4; i >= 0; --i)
        {
            top_preds[preds_offset + i] = q[idx].top().second;
            q[idx].pop();
        }
    }
}

void ctopk(float* scores, long* top_preds, int B, int N, int K, int n_threads)
{       
    vector<priority_queue<pair<float, long>>> q(N);
    #pragma omp parallel for num_threads(n_threads)
    for(int idx = 0; idx < N; ++idx)
    {   
        const int offset = idx * B;
        const int preds_offset = idx * K;
        // filling the queue
        for(long i = 0; i < B; ++i)
        {
            if(q[idx].size()<K)
                q[idx].push(pair<float, long>(-scores[offset + i], i));
            else if(q[idx].top().first > -scores[offset + i]){
                q[idx].pop();
                q[idx].push(pair<float, long>(-scores[offset + i], i));
            }    
        }
        // getting the top K classes
        for(long i = K-1; i >= 0; --i)
        {
            top_preds[preds_offset + i] = q[idx].top().second;
            q[idx].pop();
        }
    }
}

void ctopk_csr(long* indices, long* indptr, float* data, long* top_preds, int batch_size, int topk, int n_threads)
{       
    vector<priority_queue<pair<float, long>>> q(batch_size);
    #pragma omp parallel for num_threads(n_threads)
    for(int idx = 0; idx < batch_size; ++idx)
    {   
        const int preds_offset = idx * topk;
        for(long i = indptr[idx]; i < indptr[idx+1]; ++i)
        {
            if(q[idx].size()<topk)
                q[idx].push(pair<float, long>(-data[i], indices[i]));
            else if(q[idx].top().first > -data[i]){
                q[idx].pop();
                q[idx].push(pair<float, long>(-data[i], indices[i]));
            }
        }
        // getting the top-topk classes
        for(long i = topk-1; i >= 0; --i)
        {
            top_preds[preds_offset + i] = q[idx].top().second;
            q[idx].pop();
        }
    }
}
