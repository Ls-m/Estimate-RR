#include <ATen/cuda/CUDAContext.h>
#include <iostream>

// The CUDA kernel for the forward pass
__global__ void wkv_forward_kernel(const int B, const int T, const int C, const float *w, const float *u, const float *k, const float *v, float *y) {
    const int b = blockIdx.x;
    const int c = blockIdx.y;
    const int idx = b * C + c;

    float state = 0;
    const float *k_ptr = k + idx * T;
    const float *v_ptr = v + idx * T;
    float *y_ptr = y + idx * T;

    const float u_val = u[c];
    const float w_val = w[c];

    for (int t = 0; t < T; t++) {
        const float kt = k_ptr[t];
        const float vt = v_ptr[t];
        const float wkv = (t == 0) ? (u_val + kt) : (expf(w_val) * state + kt);
        y_ptr[t] = wkv * vt;
        state = (t == 0) ? kt : (expf(-expf(w_val)) * state + kt);
    }
}

// Simplified kernel for RWKV-v5 style forward
__global__ void wkv5_forward_kernel(const int B, const int T, const int C, const float *w, const float *u, const float *k, const float *v, float *y) {
    const int b = blockIdx.x;
    const int c = blockIdx.y;
    const int idx = b * C + c;

    float alpha = 0;
    float beta = 0;
    float gamma = -1e38;

    const float *k_ptr = k + idx * T;
    const float *v_ptr = v + idx * T;
    float *y_ptr = y + idx * T;

    const float u_val = u[c];
    const float w_val = -expf(w[c]);

    for (int t = 0; t < T; t++) {
        const float kt = k_ptr[t];
        const float vt = v_ptr[t];
        
        float max_gamma = fmaxf(gamma, u_val + kt);
        alpha = expf(gamma - max_gamma) * alpha + expf(u_val + kt - max_gamma) * vt;
        beta = expf(gamma - max_gamma) * beta + expf(u_val + kt - max_gamma);
        gamma = max_gamma;

        y_ptr[t] = alpha / beta;

        max_gamma = fmaxf(gamma, w_val);
        alpha = expf(gamma - max_gamma) * alpha;
        beta = expf(gamma - max_gamma) * beta;
        gamma = max_gamma;
    }
}


// The CUDA kernel for the backward pass
__global__ void wkv_backward_kernel(const int B, const int T, const int C, const float *w, const float *u, const float *k, const float *v, const float *gy, float *gw, float *gu, float *gk, float *gv) {
    const int b = blockIdx.x;
    const int c = blockIdx.y;
    const int idx = b * C + c;

    float state = 0;
    float g_state = 0;

    const float *k_ptr = k + idx * T;
    const float *v_ptr = v + idx * T;
    const float *gy_ptr = gy + idx * T;
    float *gk_ptr = gk + idx * T;
    float *gv_ptr = gv + idx * T;

    const float u_val = u[c];
    const float w_val = w[c];
    const float ew = expf(w_val);
    const float eew = expf(-expf(w_val));

    for (int t = T - 1; t >= 0; t--) {
        const float kt = k_ptr[t];
        const float vt = v_ptr[t];
        const float gyt = gy_ptr[t];

        const float wkv = (t == 0) ? (u_val + kt) : (ew * state + kt);
        gv_ptr[t] = gyt * wkv;

        float g_wkv = gyt * vt;
        gk_ptr[t] = g_wkv;

        if (t > 0) {
            g_state += g_wkv * ew;
        } else {
            atomicAdd(gu + c, g_wkv);
        }

        if (t > 0) {
            atomicAdd(gw + c, g_state * state * ew);
        }
        
        gk_ptr[t] += g_state;
        state = (t == 0) ? kt : (eew * state + kt);
        g_state = g_state * eew;
    }
}


extern "C" {
    void wkv_forward(int B, int T, int C, float *w, float *u, float *k, float *v, float *y) {
        dim3 blocks(B, C);
        dim3 threads(1);
        wkv5_forward_kernel<<<blocks, threads>>>(B, T, C, w, u, k, v, y);
    }

    void wkv_backward(int B, int T, int C, float *w, float *u, float *k, float *v, float *gy, float *gw, float *gu, float *gk, float *gv) {
        dim3 blocks(B, C);
        dim3 threads(1);
        wkv_backward_kernel<<<blocks, threads>>>(B, T, C, w, u, k, v, gy, gw, gu, gk, gv);
    }
}  