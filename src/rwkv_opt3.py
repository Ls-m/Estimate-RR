import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline
from typing import Optional, List, Tuple
import math

cuda_kernel_source = '''
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

__global__ void wkv_forward_kernel(
    const float* __restrict__ w,
    const float* __restrict__ u,
    const float* __restrict__ k,
    const float* __restrict__ v,
    const float* __restrict__ last_state,
    float* __restrict__ y,
    float* __restrict__ new_state,
    float* __restrict__ aa_all,
    float* __restrict__ bb_all,
    float* __restrict__ pp_all,
    float* __restrict__ aa_next_all,
    float* __restrict__ bb_next_all,
    float* __restrict__ pp_next_all,
    const int B, const int T, const int C
) {
    const int b = blockIdx.x;
    const int c = threadIdx.x;
    
    if (b >= B || c >= C) return;
    
    // Load initial state - FIXED bb initialization
    float aa = (last_state != nullptr) ? last_state[b * C * 3 + c * 3 + 0] : 0.0f;
    float bb = (last_state != nullptr) ? last_state[b * C * 3 + c * 3 + 1] : 1.0f;  // FIXED: Initialize to 1.0f
    float pp = (last_state != nullptr) ? last_state[b * C * 3 + c * 3 + 2] : -1e38f;
    
    const float ww = w[c];
    const float uu = u[c];
    
    for (int t = 0; t < T; t++) {
        const int idx = b * T * C + t * C + c;
        const float kk = k[idx];
        const float vv = v[idx];
        
        // Store states BEFORE computing output
        aa_all[idx] = aa;
        bb_all[idx] = bb;
        pp_all[idx] = pp;
        
        // === WKV COMPUTATION ===
        const float uu_kk = uu + kk;
        const float p_out = fmaxf(pp, uu_kk);
        const float e1 = expf(pp - p_out);
        const float e2 = expf(uu_kk - p_out);
        
        // Compute weighted sum with correct normalization
        const float num = e1 * aa + e2 * vv;
        const float den = e1 * bb + e2;
        y[idx] = num / den;
        
        // === STATE UPDATE ===
        const float ww_kk = ww + kk;
        const float p_update = fmaxf(pp + ww, ww_kk);
        const float e1_update = expf(pp + ww - p_update);
        const float e2_update = expf(ww_kk - p_update);
        
        const float aa_next = e1_update * aa + e2_update * vv;
        const float bb_next = e1_update * bb + e2_update;
        const float pp_next = p_update + logf(bb_next);
        
        // Store updated states
        aa_next_all[idx] = aa_next;
        bb_next_all[idx] = bb_next;
        pp_next_all[idx] = pp_next;
        
        // Update for next timestep
        aa = aa_next;
        bb = bb_next;
        pp = pp_next;
    }
    
    // Store final state
    new_state[b * C * 3 + c * 3 + 0] = aa;
    new_state[b * C * 3 + c * 3 + 1] = bb;
    new_state[b * C * 3 + c * 3 + 2] = pp;
}

__global__ void wkv_backward_kernel(
    const float* __restrict__ w,
    const float* __restrict__ u,
    const float* __restrict__ k,
    const float* __restrict__ v,
    const float* __restrict__ aa_all,
    const float* __restrict__ bb_all,
    const float* __restrict__ pp_all,
    const float* __restrict__ aa_next_all,
    const float* __restrict__ bb_next_all,
    const float* __restrict__ pp_next_all,
    const float* __restrict__ gy,
    float* __restrict__ gw,
    float* __restrict__ gu,
    float* __restrict__ gk,
    float* __restrict__ gv,
    float* __restrict__ gstate,
    const bool has_state,
    const int B, const int T, const int C
) {
    const int b = blockIdx.x;
    const int c = threadIdx.x;
    
    if (b >= B || c >= C) return;
    
    const float ww = w[c];
    const float uu = u[c];
    
    // Gradients flowing backward through recurrent state
    float gaa = 0.0f;
    float gbb = 0.0f;
    float gpp = 0.0f;
    
    float gw_acc = 0.0f;
    float gu_acc = 0.0f;
    
    // BACKWARD PASS: Reverse time order
    for (int t = T - 1; t >= 0; t--) {
        const int idx = b * T * C + t * C + c;
        
        const float grad_y = gy[idx];
        const float kk = k[idx];
        const float vv = v[idx];
        
        // Get stored forward values
        const float aa = aa_all[idx];
        const float bb = bb_all[idx];
        const float pp = pp_all[idx];
        
        // === BACKWARD THROUGH WKV OUTPUT ===
        const float uu_kk = uu + kk;
        const float p_out = fmaxf(pp, uu_kk);
        const float e1 = expf(pp - p_out);
        const float e2 = expf(uu_kk - p_out);
        
        const float num = e1 * aa + e2 * vv;
        const float den = e1 * bb + e2;
        
        // Gradients of y = num / den
        const float dy_dnum = 1.0f / den;
        const float dy_dden = -num / (den * den);
        
        // Gradients w.r.t. exponentials in output
        const float dnum_de1 = aa;
        const float dnum_de2 = vv;
        const float dden_de1 = bb;
        const float dden_de2 = 1.0f;
        
        const float dy_de1 = dy_dnum * dnum_de1 + dy_dden * dden_de1;
        const float dy_de2 = dy_dnum * dnum_de2 + dy_dden * dden_de2;
        
        // Gradients w.r.t. exponential arguments
        const float de1_dpp = e1;
        const float de1_dp_out = -e1;
        const float de2_duu_kk = e2;
        const float de2_dp_out = -e2;
        
        // Gradients w.r.t. p_out = max(pp, uu_kk)
        const float dp_out_dpp = (pp >= uu_kk) ? 1.0f : 0.0f;
        const float dp_out_duu_kk = (pp >= uu_kk) ? 0.0f : 1.0f;
        
        // Chain rule for current timestep
        const float grad_e1 = grad_y * dy_de1;
        const float grad_e2 = grad_y * dy_de2;
        const float grad_p_out = grad_e1 * de1_dp_out + grad_e2 * de2_dp_out;
        
        // Direct gradients w.r.t. inputs
        gv[idx] = grad_y * dy_dnum * dnum_de2;  // From vv in numerator
        
        // Gradient w.r.t. uu_kk (affects both k and u)
        const float grad_uu_kk = grad_e2 * de2_duu_kk + grad_p_out * dp_out_duu_kk;
        gk[idx] = grad_uu_kk;  // d(uu_kk)/dk = 1
        gu_acc += grad_uu_kk;  // d(uu_kk)/du = 1
        
        // Gradients w.r.t. previous states (from output computation)
        gaa += grad_y * dy_dnum * dnum_de1;  // From aa in numerator
        gbb += grad_y * dy_dden * dden_de1;  // From bb in denominator
        gpp += grad_e1 * de1_dpp + grad_p_out * dp_out_dpp;
        
        // === BACKWARD THROUGH STATE UPDATE ===
        if (t < T - 1) {  // If there's a next timestep
            // Current gradients from next timestep
            const float gaa_from_next = gaa;
            const float gbb_from_next = gbb;
            const float gpp_from_next = gpp;
            
            // Get state update values
            const float ww_kk = ww + kk;
            const float p_update = fmaxf(pp + ww, ww_kk);
            const float e1_update = expf(pp + ww - p_update);
            const float e2_update = expf(ww_kk - p_update);
            const float bb_next = e1_update * bb + e2_update;
            
            // Gradients w.r.t. update exponentials
            const float grad_e1_update = gaa_from_next * aa + gbb_from_next * bb;
            const float grad_e2_update = gaa_from_next * vv + gbb_from_next * 1.0f;
            
            // From pp_next gradient
            const float grad_p_update_direct = gpp_from_next * 1.0f;
            const float grad_bb_next = gpp_from_next * (1.0f / bb_next);
            
            const float grad_e1_update_total = grad_e1_update + grad_bb_next * bb;
            const float grad_e2_update_total = grad_e2_update + grad_bb_next * 1.0f;
            
            // Exponential derivatives
            const float de1_update_dpp_ww = e1_update;
            const float de1_update_dp_update = -e1_update;
            const float de2_update_dww_kk = e2_update;
            const float de2_update_dp_update = -e2_update;
            
            const float grad_p_update_total = grad_p_update_direct + 
                                            grad_e1_update_total * de1_update_dp_update + 
                                            grad_e2_update_total * de2_update_dp_update;
            
            // p_update = max(pp + ww, ww_kk)
            const float dp_update_dpp_ww = (pp + ww >= ww_kk) ? 1.0f : 0.0f;
            const float dp_update_dww_kk = (pp + ww >= ww_kk) ? 0.0f : 1.0f;
            
            // Gradients w.r.t. w
            const float grad_pp_ww = grad_e1_update_total * de1_update_dpp_ww + 
                                   grad_p_update_total * dp_update_dpp_ww;
            const float grad_ww_kk_from_update = grad_e2_update_total * de2_update_dww_kk + 
                                                grad_p_update_total * dp_update_dww_kk;
            
            gw_acc += grad_pp_ww + grad_ww_kk_from_update;
            gk[idx] += grad_ww_kk_from_update;  // d(ww_kk)/dk = 1
            
            // Additional v gradient from state update
            gv[idx] += gaa_from_next * e2_update;
            
            // Update gradients for previous timestep
            gaa = gaa_from_next * e1_update;
            gbb = gbb_from_next * e1_update;
            gpp = grad_pp_ww + grad_e1_update_total * de1_update_dpp_ww + grad_p_update_total * dp_update_dpp_ww;
        }
    }
    
    // Accumulate gradients
    atomicAdd(&gw[c], gw_acc);
    atomicAdd(&gu[c], gu_acc);
    
    // Set initial state gradients
    if (has_state && gstate != nullptr) {
        gstate[b * C * 3 + c * 3 + 0] = gaa;
        gstate[b * C * 3 + c * 3 + 1] = gbb;
        gstate[b * C * 3 + c * 3 + 2] = gpp;
    }
}

std::vector<torch::Tensor> wkv_cuda_forward(
    torch::Tensor w,
    torch::Tensor u,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor last_state
) {
    const int B = k.size(0);
    const int T = k.size(1);
    const int C = k.size(2);
    
    auto y = torch::zeros_like(v);
    auto new_state = torch::zeros({B, C, 3}, k.options());
    
    // Store ALL intermediate values for exact backward computation
    auto aa_all = torch::zeros({B, T, C}, k.options());
    auto bb_all = torch::zeros({B, T, C}, k.options());
    auto pp_all = torch::zeros({B, T, C}, k.options());
    auto aa_next_all = torch::zeros({B, T, C}, k.options());
    auto bb_next_all = torch::zeros({B, T, C}, k.options());
    auto pp_next_all = torch::zeros({B, T, C}, k.options());
    
    const int block_size = std::min(C, 1024);
    const dim3 blocks(B);
    const dim3 threads(block_size);
    
    wkv_forward_kernel<<<blocks, threads>>>(
        w.data_ptr<float>(),
        u.data_ptr<float>(),
        k.data_ptr<float>(),
        v.data_ptr<float>(),
        last_state.numel() > 0 ? last_state.data_ptr<float>() : nullptr,
        y.data_ptr<float>(),
        new_state.data_ptr<float>(),
        aa_all.data_ptr<float>(),
        bb_all.data_ptr<float>(),
        pp_all.data_ptr<float>(),
        aa_next_all.data_ptr<float>(),
        bb_next_all.data_ptr<float>(),
        pp_next_all.data_ptr<float>(),
        B, T, C
    );
    
    cudaDeviceSynchronize();
    return {y, new_state, aa_all, bb_all, pp_all, aa_next_all, bb_next_all, pp_next_all};
}

std::vector<torch::Tensor> wkv_cuda_backward(
    torch::Tensor w,
    torch::Tensor u,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor aa_all,
    torch::Tensor bb_all,
    torch::Tensor pp_all,
    torch::Tensor aa_next_all,
    torch::Tensor bb_next_all,
    torch::Tensor pp_next_all,
    torch::Tensor grad_y,
    bool has_state
) {
    const int B = k.size(0);
    const int T = k.size(1);
    const int C = k.size(2);
    
    auto grad_w = torch::zeros_like(w);
    auto grad_u = torch::zeros_like(u);
    auto grad_k = torch::zeros_like(k);
    auto grad_v = torch::zeros_like(v);
    auto grad_state = has_state ? torch::zeros({B, C, 3}, k.options()) : torch::empty(0);
    
    const int block_size = std::min(C, 1024);
    const dim3 blocks(B);
    const dim3 threads(block_size);
    
    wkv_backward_kernel<<<blocks, threads>>>(
        w.data_ptr<float>(),
        u.data_ptr<float>(),
        k.data_ptr<float>(),
        v.data_ptr<float>(),
        aa_all.data_ptr<float>(),
        bb_all.data_ptr<float>(),
        pp_all.data_ptr<float>(),
        aa_next_all.data_ptr<float>(),
        bb_next_all.data_ptr<float>(),
        pp_next_all.data_ptr<float>(),
        grad_y.data_ptr<float>(),
        grad_w.data_ptr<float>(),
        grad_u.data_ptr<float>(),
        grad_k.data_ptr<float>(),
        grad_v.data_ptr<float>(),
        has_state ? grad_state.data_ptr<float>() : nullptr,
        has_state,
        B, T, C
    );
    
    cudaDeviceSynchronize();
    return {grad_w, grad_u, grad_k, grad_v, grad_state};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("wkv_cuda_forward", &wkv_cuda_forward, "WKV forward (CUDA)");
    m.def("wkv_cuda_backward", &wkv_cuda_backward, "WKV backward (CUDA)");
}
'''

# FIXED CPU implementation - consistent bb initialization
def wkv_cpu_forward_correct(w, u, k, v, last_state):
    """Mathematically correct CPU implementation - FIXED bb initialization."""
    B, T, C = k.size()
    device = k.device
    dtype = k.dtype
    
    y = torch.zeros_like(v)
    new_state = torch.zeros(B, C, 3, device=device, dtype=dtype)
    
    # Store ALL values needed for exact backward pass
    aa_all = torch.zeros(B, T, C, device=device, dtype=dtype)
    bb_all = torch.zeros(B, T, C, device=device, dtype=dtype)
    pp_all = torch.zeros(B, T, C, device=device, dtype=dtype)
    aa_next_all = torch.zeros(B, T, C, device=device, dtype=dtype)
    bb_next_all = torch.zeros(B, T, C, device=device, dtype=dtype)
    pp_next_all = torch.zeros(B, T, C, device=device, dtype=dtype)
    
    for b in range(B):
        if last_state is not None and last_state.numel() > 0:
            aa = last_state[b, :, 0].clone()
            bb = last_state[b, :, 1].clone()
            pp = last_state[b, :, 2].clone()
        else:
            aa = torch.zeros(C, device=device, dtype=dtype)
            bb = torch.ones(C, device=device, dtype=dtype)  # CONSISTENT: Initialize bb to 1
            pp = torch.full((C,), -1e38, device=device, dtype=dtype)
        
        for t in range(T):
            # Store states BEFORE computation
            aa_all[b, t] = aa.clone()
            bb_all[b, t] = bb.clone()
            pp_all[b, t] = pp.clone()
            
            kk = k[b, t]
            vv = v[b, t]
            
            # === CORRECT WKV COMPUTATION ===
            uu_kk = u + kk
            p_out = torch.maximum(pp, uu_kk)
            e1 = torch.exp(pp - p_out)
            e2 = torch.exp(uu_kk - p_out)
            
            # Weighted sum with correct denominator
            num = e1 * aa + e2 * vv
            den = e1 * bb + e2  # bb is the normalization factor
            y[b, t] = num / den
            
            # === CORRECT STATE UPDATE ===
            ww_kk = w + kk
            p_update = torch.maximum(pp + w, ww_kk)
            e1_update = torch.exp(pp + w - p_update)
            e2_update = torch.exp(ww_kk - p_update)
            
            aa_next = e1_update * aa + e2_update * vv
            bb_next = e1_update * bb + e2_update
            pp_next = p_update + torch.log(bb_next)
            
            # Store updated states
            aa_next_all[b, t] = aa_next.clone()
            bb_next_all[b, t] = bb_next.clone()
            pp_next_all[b, t] = pp_next.clone()
            
            # Update for next timestep
            aa = aa_next
            bb = bb_next
            pp = pp_next
        
        new_state[b, :, 0] = aa
        new_state[b, :, 1] = bb
        new_state[b, :, 2] = pp
    
    return y, new_state, aa_all, bb_all, pp_all, aa_next_all, bb_next_all, pp_next_all

def wkv_cpu_backward_correct(w, u, k, v, aa_all, bb_all, pp_all, aa_next_all, bb_next_all, pp_next_all, grad_y, has_state):
    """Mathematically correct CPU backward pass."""
    B, T, C = k.size()
    device = k.device
    dtype = k.dtype
    
    grad_w = torch.zeros_like(w)
    grad_u = torch.zeros_like(u)
    grad_k = torch.zeros_like(k)
    grad_v = torch.zeros_like(v)
    
    if has_state:
        grad_state = torch.zeros(B, C, 3, device=device, dtype=dtype)
    else:
        grad_state = None
    
    for b in range(B):
        # Initialize gradients flowing backward through recurrent connections
        gaa = torch.zeros(C, device=device, dtype=dtype)
        gbb = torch.zeros(C, device=device, dtype=dtype)
        gpp = torch.zeros(C, device=device, dtype=dtype)
        
        # Backward pass through time
        for t in reversed(range(T)):
            grad_output = grad_y[b, t]
            kk = k[b, t]
            vv = v[b, t]
            
            # Get stored forward values
            aa = aa_all[b, t]
            bb = bb_all[b, t]
            pp = pp_all[b, t]
            
            # === BACKWARD THROUGH WKV OUTPUT ===
            uu_kk = u + kk
            p_out = torch.maximum(pp, uu_kk)
            e1 = torch.exp(pp - p_out)
            e2 = torch.exp(uu_kk - p_out)
            
            num = e1 * aa + e2 * vv
            den = e1 * bb + e2
            
            # Gradients of y = num / den using quotient rule
            dy_dnum = 1.0 / den
            dy_dden = -num / (den * den)
            
            # Gradients w.r.t. exponentials
            dnum_de1 = aa
            dnum_de2 = vv
            dden_de1 = bb
            dden_de2 = torch.ones_like(e2)
            
            dy_de1 = dy_dnum * dnum_de1 + dy_dden * dden_de1
            dy_de2 = dy_dnum * dnum_de2 + dy_dden * dden_de2
            
            # Gradients w.r.t. exponential arguments
            de1_dpp = e1
            de1_dp_out = -e1
            de2_duu_kk = e2
            de2_dp_out = -e2
            
            # Max operation gradients
            pp_mask = (pp >= uu_kk).float()
            uu_kk_mask = 1.0 - pp_mask
            dp_out_dpp = pp_mask
            dp_out_duu_kk = uu_kk_mask
            
            # Chain rule
            grad_e1 = grad_output * dy_de1
            grad_e2 = grad_output * dy_de2
            grad_p_out = grad_e1 * de1_dp_out + grad_e2 * de2_dp_out
            
            # Gradients w.r.t. inputs
            grad_v[b, t] += grad_output * dy_dnum * dnum_de2  # Direct from vv in numerator
            
            grad_uu_kk = grad_e2 * de2_duu_kk + grad_p_out * dp_out_duu_kk
            grad_k[b, t] += grad_uu_kk  # d(uu_kk)/dk = 1
            grad_u += grad_uu_kk        # d(uu_kk)/du = 1
            
            # Gradients w.r.t. states from output
            gaa += grad_output * dy_dnum * dnum_de1
            gbb += grad_output * dy_dden * dden_de1
            gpp += grad_e1 * de1_dpp + grad_p_out * dp_out_dpp
            
            # === BACKWARD THROUGH STATE UPDATE ===
            if t < T - 1:  # Not the last timestep
                # Get gradients from next timestep
                gaa_next = gaa.clone()
                gbb_next = gbb.clone()
                gpp_next = gpp.clone()
                
                # State update equations (repeated for gradient computation)
                ww_kk = w + kk
                p_update = torch.maximum(pp + w, ww_kk)
                e1_update = torch.exp(pp + w - p_update)
                e2_update = torch.exp(ww_kk - p_update)
                bb_next = e1_update * bb + e2_update
                
                # Gradients w.r.t. updated states
                grad_e1_update = gaa_next * aa + gbb_next * bb
                grad_e2_update = gaa_next * vv + gbb_next * torch.ones_like(vv)
                
                # From pp_next gradient
                grad_p_update_from_pp = gpp_next * torch.ones_like(gpp_next)
                grad_bb_next_from_pp = gpp_next / bb_next  # Safe division
                
                grad_e1_update += grad_bb_next_from_pp * bb
                grad_e2_update += grad_bb_next_from_pp * torch.ones_like(grad_e2_update)
                
                # Exponential gradients
                de1_update_dpp_ww = e1_update
                de1_update_dp_update = -e1_update
                de2_update_dww_kk = e2_update
                de2_update_dp_update = -e2_update
                
                grad_p_update = grad_p_update_from_pp + grad_e1_update * de1_update_dp_update + grad_e2_update * de2_update_dp_update
                
                # Max operation in state update
                pp_w_mask = (pp + w >= ww_kk).float()
                ww_kk_mask = 1.0 - pp_w_mask
                
                # Gradients to w
                grad_pp_ww = grad_e1_update * de1_update_dpp_ww + grad_p_update * pp_w_mask
                grad_ww_kk_update = grad_e2_update * de2_update_dww_kk + grad_p_update * ww_kk_mask
                
                grad_w += grad_pp_ww + grad_ww_kk_update
                grad_k[b, t] += grad_ww_kk_update
                
                # Additional v gradient from state update
                grad_v[b, t] += gaa_next * e2_update
                
                # Update gradients for previous timestep
                gaa = gaa_next * e1_update
                gbb = gbb_next * e1_update
                gpp = grad_pp_ww + grad_e1_update * de1_update_dpp_ww + grad_p_update * pp_w_mask
        
        # Store initial state gradients
        if has_state and grad_state is not None:
            grad_state[b, :, 0] = gaa
            grad_state[b, :, 1] = gbb
            grad_state[b, :, 2] = gpp
    
    return grad_w, grad_u, grad_k, grad_v, grad_state


class MathematicallyCorrectWKV(torch.autograd.Function):
    """Mathematically correct WKV - FIXED gradient flow."""
    
    @staticmethod
    def forward(ctx, w, u, k, v, last_state):
        # REMOVED .detach() calls that were breaking gradient flow
        w = w.contiguous()
        u = u.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        
        has_state = last_state is not None and last_state.numel() > 0
        
        if has_state:
            last_state = last_state.contiguous()
        
        if CUDA_AVAILABLE and k.is_cuda:
            y, new_state, aa_all, bb_all, pp_all, aa_next_all, bb_next_all, pp_next_all = wkv_cuda.wkv_cuda_forward(w, u, k, v, last_state if has_state else torch.empty(0, device=k.device, dtype=k.dtype))
        else:
            y, new_state, aa_all, bb_all, pp_all, aa_next_all, bb_next_all, pp_next_all = wkv_cpu_forward_correct(w, u, k, v, last_state if has_state else None)
        
        # Save all necessary tensors for backward
        if has_state:
            ctx.save_for_backward(w, u, k, v, last_state, aa_all, bb_all, pp_all, aa_next_all, bb_next_all, pp_next_all)
        else:
            placeholder_state = torch.empty(0, device=w.device, dtype=w.dtype)
            ctx.save_for_backward(w, u, k, v, placeholder_state, aa_all, bb_all, pp_all, aa_next_all, bb_next_all, pp_next_all)
        
        ctx.has_state = has_state
        
        return y, new_state
    
    @staticmethod
    def backward(ctx, grad_y, grad_new_state):
        w, u, k, v, last_state_or_placeholder, aa_all, bb_all, pp_all, aa_next_all, bb_next_all, pp_next_all = ctx.saved_tensors
        
        grad_y = grad_y.contiguous()
        
        if CUDA_AVAILABLE and k.is_cuda:
            grad_w, grad_u, grad_k, grad_v, grad_last_state = wkv_cuda.wkv_cuda_backward(
                w, u, k, v, aa_all, bb_all, pp_all, aa_next_all, bb_next_all, pp_next_all, grad_y, ctx.has_state
            )
        else:
            grad_w, grad_u, grad_k, grad_v, grad_last_state = wkv_cpu_backward_correct(
                w, u, k, v, aa_all, bb_all, pp_all, aa_next_all, bb_next_all, pp_next_all, grad_y, ctx.has_state
            )
        
        if not ctx.has_state:
            grad_last_state = None
        
        return grad_w, grad_u, grad_k, grad_v, grad_last_state


# Try to compile CUDA kernel
CUDA_AVAILABLE = False
try:
    wkv_cuda = load_inline(
        name='wkv_fixed_gradient_flow',
        cpp_sources='',
        cuda_sources=cuda_kernel_source,
        verbose=True,
        extra_cflags=['-O3'],
        extra_cuda_cflags=['-O3', '--use_fast_math', '--expt-relaxed-constexpr']
    )
    CUDA_AVAILABLE = True
    print("CUDA kernel with fixed gradient flow compiled successfully")
except Exception as e:
    print(f"CUDA kernel compilation failed: {e}")
    print("Falling back to CPU implementation")


class RWKVBlock(nn.Module):
    """Single RWKV block with time mixing and channel mixing."""
    
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        
        # Time mixing parameters
        self.time_decay = nn.Parameter(torch.randn(d_model))
        self.time_first = nn.Parameter(torch.randn(d_model))
        
        # Time mixing layers
        self.time_mix_k = nn.Linear(d_model, d_model, bias=False)
        self.time_mix_v = nn.Linear(d_model, d_model, bias=False)
        self.time_mix_r = nn.Linear(d_model, d_model, bias=False)
        
        # Channel mixing layers
        self.channel_mix_k = nn.Linear(d_model, d_model * 4, bias=False)
        self.channel_mix_v = nn.Linear(d_model * 4, d_model, bias=False)
        self.channel_mix_r = nn.Linear(d_model, d_model, bias=False)
        
        # Layer normalization
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Time shift parameters
        self.time_shift = nn.Parameter(torch.zeros(1, 1, d_model))
        
    def time_mixing(self, x: torch.Tensor, state: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Time mixing operation."""
        B, T, C = x.size()
        
        # Time shift
        if T > 1:
            x_shifted = torch.cat([self.time_shift.expand(B, 1, C), x[:, :-1, :]], dim=1)
        else:
            x_shifted = self.time_shift.expand(B, T, C)
        
        # Mix current and previous states
        xx = x - x_shifted
        xxx = x + xx * self.time_shift
        
        # Compute key, value, receptance
        k = self.time_mix_k(xxx)
        v = self.time_mix_v(xxx)
        r = self.time_mix_r(xxx)
        
        # Apply sigmoid to receptance
        r = torch.sigmoid(r)
        
        # WKV operation using optimized kernel
        w = -torch.exp(self.time_decay)
        u = self.time_first
        
        wkv_out, new_state = MathematicallyCorrectWKV.apply(w, u, k, v, state)
        
        return r * wkv_out, new_state
    
    def channel_mixing(self, x: torch.Tensor) -> torch.Tensor:
        """Channel mixing operation."""
        # Time shift for channel mixing
        if x.size(1) > 1:
            xx = torch.cat([x[:, -1:, :], x[:, :-1, :]], dim=1)
        else:
            xx = x
        xxx = x + (xx - x) * self.time_shift
        
        k = self.channel_mix_k(xxx)
        r = self.channel_mix_r(xxx)
        
        # Apply activation and channel mixing
        vv = self.channel_mix_v(F.relu(k) ** 2)
        
        return torch.sigmoid(r) * vv
    
    def forward(self, x: torch.Tensor, state: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through RWKV block."""
        # Time mixing
        tm_out, new_state = self.time_mixing(self.ln1(x), state)
        x = x + self.dropout(tm_out)
        
        # Channel mixing
        cm_out = self.channel_mixing(self.ln2(x))
        x = x + self.dropout(cm_out)
        
        return x, new_state


class RWKV(nn.Module):
    """Multi-layer RWKV model for time series."""

    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Input projection
        self.input_proj = nn.Linear(input_size, hidden_size)
        
        # RWKV blocks
        self.blocks = nn.ModuleList([
            RWKVBlock(hidden_size, dropout) for _ in range(num_layers)
        ])
        
        # Output layer norm
        self.ln_out = nn.LayerNorm(hidden_size)
        
        # Initialize parameters
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    def forward(self, x: torch.Tensor, state: Optional[List[torch.Tensor]] = None) -> torch.Tensor:
        """Forward pass through RWKV."""
        # Input projection
        x = self.input_proj(x)
        
        # Process through RWKV blocks
        new_states = []
        if state is None:
            state = [None] * self.num_layers
        for i, block in enumerate(self.blocks):
            x, block_new_state = block(x, state[i])
            new_states.append(block_new_state)
        
        # Output normalization
        x = self.ln_out(x)
        
        # Return final representation (last time step)
        return x[:, -1, :]  # (batch_size, hidden_size)
    

class RWKVRRModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=64, dropout=0.2):
        super().__init__()
        self.rwkv = RWKV(input_size, hidden_size, num_layers, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x: (B, T, 1)
        rwkv_out = self.rwkv(x)        # returns (B, hidden_size)
        out = self.fc(rwkv_out)        # map to embedding dimension
        return out