import triton.language as tl
import torch
import os
import triton
from torch._inductor.async_compile import AsyncCompile
import triton_kernels_benchmark as benchmark_suit

async_compile = AsyncCompile()

is_cuda_available = torch.cuda.is_available()
is_xpu_available = torch.xpu.is_available()
if is_cuda_available:
    CUDA_CAPABILITY = torch.cuda.get_device_capability()

USE_BLOCK_PTR = int(os.getenv('USE_BLOCK_PTR', 0))


@triton.jit
def tanh(x):
    # Tanh is just a scaled sigmoid
    return 2 * tl.sigmoid(2 * x) - 1


@triton.jit
def _attn_fwd_inner(acc, deno, e_max, q, k_block_ptr, v_block_ptr, start_m, sm_scale, logit_cap, cur_block_m_end,
                    mask_m, BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr, BLOCK_N: tl.constexpr,
                    STAGE: tl.constexpr, offs_m: tl.constexpr, offs_n: tl.constexpr):
    # range of values handled by this stage
    if STAGE == 1:
        lo, hi = 0, start_m * BLOCK_M
    elif STAGE == 2:
        lo, hi = start_m * BLOCK_M, (start_m + 1) * BLOCK_M
        lo = tl.multiple_of(lo, BLOCK_M)
        hi = cur_block_m_end
    # causal = False
    else:
        lo, hi = 0, cur_block_m_end
    k_block_ptr = tl.advance(k_block_ptr, (0, lo))
    v_block_ptr = tl.advance(v_block_ptr, (lo, 0))

    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        mask_n = (start_n + offs_n) < cur_block_m_end

        # k:[BLOCK_N, 1, BLOCK_DMODEL]
        k = tl.load(k_block_ptr, boundary_check=(0, 1))
        # k = k.trans()

        qk = tl.dot(q, k, out_dtype=tl.float32)
        qk *= sm_scale

        if logit_cap > 0:
            qk = logit_cap * tanh(qk / logit_cap)

        if STAGE == 2:
            # mask_causual = (cur_block_m * BLOCK_M + offs_m[:, None]) >= (
            #    start_n + offs_n[None, :]
            # )
            # mask_causual &= mask_m[:, None] & mask_n[None, :]
            mask = offs_m[:, None] >= (start_n + offs_n[None, :])
            mask &= mask_m[:, None] & mask_n[None, :]
            qk = tl.where(mask, qk, float('-inf'))

        n_e_max = tl.maximum(tl.max(qk, 1), e_max)
        re_scale = tl.exp(e_max - n_e_max)
        p = tl.exp(qk - n_e_max[:, None])
        deno = deno * re_scale + tl.sum(p, 1)
        v = tl.load(v_block_ptr, boundary_check=(0, 1))

        p = p.to(v.dtype)
        acc = acc * re_scale[:, None] + tl.dot(p, v)

        e_max = n_e_max

        v_block_ptr = tl.advance(v_block_ptr, (BLOCK_N, 0))
        k_block_ptr = tl.advance(k_block_ptr, (0, BLOCK_N))
    return acc, deno, e_max


# pylint: disable=unused-argument
@triton.jit
def _fwd_kernel2(
    Q_Extend,
    K_Extend,
    V_Extend,
    O_Extend,
    K_Buffer,
    V_Buffer,
    Req_to_tokens,
    B_req_idx,
    B_Seq_Len,
    B_Start_Loc_Extend,
    B_Seq_Len_Extend,
    sm_scale,
    q_head_num,
    stride_qbs,
    stride_qh,
    stride_kbs,
    stride_kh,
    stride_vbs,
    stride_vh,
    stride_obs,
    stride_oh,
    stride_buf_kbs,
    stride_buf_kh,
    stride_buf_vbs,
    stride_buf_vh,
    stride_req_to_tokens_b,
    logit_cap: tl.constexpr,
    Lq: tl.constexpr,
    Lv: tl.constexpr,
    kv_group_num: tl.constexpr,
    use_block_ptr: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DPE: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    cur_seq = tl.program_id(2)
    cur_head = tl.program_id(0)
    cur_block_m = tl.program_id(1)
    cur_kv_head = cur_head // kv_group_num
    kv_head_num = q_head_num // kv_group_num

    cur_seq_len = tl.load(B_Seq_Len + cur_seq)
    cur_seq_len_extend = tl.load(B_Seq_Len_Extend + cur_seq)
    cur_seq_len_prefix = cur_seq_len - cur_seq_len_extend
    cur_seq_prefix_start_in_loc = 0
    cur_seq_extend_start_contiguous = tl.load(B_Start_Loc_Extend + cur_seq)
    cur_batch_req_idx = tl.load(B_req_idx + cur_seq)

    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_dv = tl.arange(0, BLOCK_DV)
    offs_m = tl.arange(0, BLOCK_M)
    mask_m = (cur_block_m * BLOCK_M + offs_m) < cur_seq_len_extend

    mask_d = offs_d < Lq
    mask_dv = offs_dv < Lv

    # Ragged inputs: Q:[S, H, D]
    offs_q = cur_seq_extend_start_contiguous * stride_qbs + cur_head * stride_qh
    # the offset to the current seq
    q_block_ptr = tl.make_block_ptr(
        base=Q_Extend + offs_q,
        shape=(cur_seq_len_extend, Lq),
        strides=(stride_qbs, 1),
        offsets=(cur_block_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0),
    )
    # q:[BLOCK_M, 1, BLOCK_DMODEL]
    q = tl.load(q_block_ptr, boundary_check=(0, 1))
    # q = tl.reshape(q, (BLOCK_M, BLOCK_DMODEL), can_reorder=False)

    if BLOCK_DPE > 0:
        offs_dpe = BLOCK_DMODEL + tl.arange(0, BLOCK_DPE)
        offs_qpe = ((cur_seq_extend_start_contiguous + cur_block_m * BLOCK_M + offs_m[:, None]) * stride_qbs +
                    cur_head * stride_qh + offs_dpe[None, :])
        qpe = tl.load(Q_Extend + offs_qpe, mask=mask_m[:, None], other=0.0)

    # stage 1: compute scores with prefix
    offs_n = tl.arange(0, BLOCK_N)

    acc = tl.zeros([BLOCK_M, BLOCK_DV], dtype=tl.float32)
    deno = tl.zeros([BLOCK_M], dtype=tl.float32)
    e_max = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')

    # stage 2: compute the trianlge part
    cur_block_m_end = tl.minimum(cur_seq_len_extend, (cur_block_m + 1) * BLOCK_M)

    offs_v = cur_seq_extend_start_contiguous * stride_vbs + cur_kv_head * stride_kh
    # the offset to the current seq
    v_block_ptr = tl.make_block_ptr(
        base=V_Extend + offs_v,
        shape=(cur_seq_len_extend, Lv),
        strides=(stride_vbs, 1),
        offsets=(0, 0),
        block_shape=(BLOCK_N, BLOCK_DMODEL),
        order=(1, 0),
    )

    offs_k = cur_seq_extend_start_contiguous * stride_kbs + cur_kv_head * stride_kh
    # the offset to the current seq
    k_block_ptr = tl.make_block_ptr(
        base=K_Extend + offs_k,
        shape=(Lv, cur_seq_len_extend),
        strides=(1, stride_kbs),
        offsets=(0, 0),
        block_shape=(BLOCK_DMODEL, BLOCK_N),
        order=(0, 1),
    )

    start_m = cur_block_m
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    STAGE = 3
    if STAGE & 1:
        acc, deno, e_max = _attn_fwd_inner(acc, deno, e_max, q, k_block_ptr, v_block_ptr,  #
                                           start_m, sm_scale, logit_cap, cur_block_m_end, mask_m, BLOCK_M, BLOCK_DMODEL,
                                           BLOCK_N,  #
                                           4 - STAGE, offs_m, offs_n  #
                                           )
    # stage 2: on-band
    if STAGE & 2:
        # barrier makes it easier for compielr to schedule the
        # two loops independently
        tl.debug_barrier()
        acc, deno, e_max = _attn_fwd_inner(acc, deno, e_max, q, k_block_ptr, v_block_ptr,  #
                                           start_m, sm_scale, logit_cap, cur_block_m_end, mask_m, BLOCK_M, BLOCK_DMODEL,
                                           BLOCK_N,  #
                                           2, offs_m, offs_n  #
                                           )

    # Ragged inputs: out:[S, H, D]
    offs_o = cur_seq_extend_start_contiguous * stride_obs + cur_head * stride_oh
    # the offset to the current seq
    o_block_ptr = tl.make_block_ptr(
        base=O_Extend + offs_o,
        shape=(cur_seq_len_extend, Lq),
        strides=(stride_obs, 1),
        offsets=(cur_block_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0),
    )
    # :[BLOCK_M, BLOCK_DMODEL]
    o = acc / deno[:, None]
    o = o.to(q.dtype)
    # o = o.reshape(BLOCK_M, 1, BLOCK_DMODEL)
    tl.store(o_block_ptr, o, boundary_check=(0, 1))


# pylint: disable=unused-argument
@triton.jit
def _fwd_kernel(
    Q_Extend,
    K_Extend,
    V_Extend,
    O_Extend,
    K_Buffer,
    V_Buffer,
    Req_to_tokens,
    B_req_idx,
    B_Seq_Len,
    B_Start_Loc_Extend,
    B_Seq_Len_Extend,
    sm_scale,
    q_head_num,
    stride_qbs,
    stride_qh,
    stride_kbs,
    stride_kh,
    stride_vbs,
    stride_vh,
    stride_obs,
    stride_oh,
    stride_buf_kbs,
    stride_buf_kh,
    stride_buf_vbs,
    stride_buf_vh,
    stride_req_to_tokens_b,
    logit_cap: tl.constexpr,
    Lq: tl.constexpr,
    Lv: tl.constexpr,
    kv_group_num: tl.constexpr,
    use_block_ptr: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DPE: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    cur_seq = tl.program_id(2)
    cur_head = tl.program_id(0)
    cur_block_m = tl.program_id(1)
    cur_kv_head = cur_head // kv_group_num
    kv_head_num = q_head_num // kv_group_num

    cur_seq_len = tl.load(B_Seq_Len + cur_seq)
    cur_seq_len_extend = tl.load(B_Seq_Len_Extend + cur_seq)
    cur_seq_len_prefix = cur_seq_len - cur_seq_len_extend
    cur_seq_prefix_start_in_loc = 0
    cur_seq_extend_start_contiguous = tl.load(B_Start_Loc_Extend + cur_seq)
    cur_batch_req_idx = tl.load(B_req_idx + cur_seq)

    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_dv = tl.arange(0, BLOCK_DV)
    offs_m = tl.arange(0, BLOCK_M)
    mask_m = (cur_block_m * BLOCK_M + offs_m) < cur_seq_len_extend

    mask_d = offs_d < Lq
    mask_dv = offs_dv < Lv
    if not use_block_ptr:
        offs_q = ((cur_seq_extend_start_contiguous + cur_block_m * BLOCK_M + offs_m[:, None]) * stride_qbs +
                  cur_head * stride_qh + offs_d[None, :])
        q = tl.load(Q_Extend + offs_q, mask=(mask_m[:, None]) & (mask_d[None, :]), other=0.0)
    else:
        # Ragged inputs: Q:[S, H, D]
        offs_q = cur_seq_extend_start_contiguous * stride_qbs + cur_head * stride_qh
        # the offset to the current seq
        q_block_ptr = tl.make_block_ptr(
            base=Q_Extend + offs_q,
            shape=(cur_seq_len_extend, Lq),
            strides=(stride_qbs, 1),
            offsets=(cur_block_m * BLOCK_M, 0),
            block_shape=(BLOCK_M, BLOCK_DMODEL),
            order=(1, 0),
        )
        # q:[BLOCK_M, 1, BLOCK_DMODEL]
        q = tl.load(q_block_ptr, boundary_check=(0, 1))
        # q = tl.reshape(q, (BLOCK_M, BLOCK_DMODEL), can_reorder=False)
    if BLOCK_DPE > 0:
        offs_dpe = BLOCK_DMODEL + tl.arange(0, BLOCK_DPE)
        offs_qpe = ((cur_seq_extend_start_contiguous + cur_block_m * BLOCK_M + offs_m[:, None]) * stride_qbs +
                    cur_head * stride_qh + offs_dpe[None, :])
        qpe = tl.load(Q_Extend + offs_qpe, mask=mask_m[:, None], other=0.0)

    # stage 1: compute scores with prefix
    offs_n = tl.arange(0, BLOCK_N)

    acc = tl.zeros([BLOCK_M, BLOCK_DV], dtype=tl.float32)
    deno = tl.zeros([BLOCK_M], dtype=tl.float32)
    e_max = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')

    # stage 2: compute the trianlge part
    cur_block_m_end = tl.minimum(cur_seq_len_extend, (cur_block_m + 1) * BLOCK_M)
    if use_block_ptr:
        offs_k = cur_seq_extend_start_contiguous * stride_kbs + cur_kv_head * stride_kh
        # the offset to the current seq
        # k_block_ptr = tl.make_block_ptr(
        #     base=K_Extend + offs_k,
        #     shape=(cur_seq_len_extend, Lv),
        #     strides=(stride_kbs, 1),
        #     offsets=(0, 0),
        #     block_shape=(BLOCK_N, BLOCK_DMODEL),
        #     order=(1, 0),
        # )
        k_block_ptr = tl.make_block_ptr(
            base=K_Extend + offs_k,
            shape=(Lv, cur_seq_len_extend),
            strides=(1, stride_kbs),
            offsets=(0, 0),
            block_shape=(BLOCK_DMODEL, BLOCK_N),
            order=(0, 1),
        )

        offs_v = cur_seq_extend_start_contiguous * stride_vbs + cur_kv_head * stride_kh
        # the offset to the current seq
        v_block_ptr = tl.make_block_ptr(
            base=V_Extend + offs_v,
            shape=(cur_seq_len_extend, Lv),
            strides=(stride_vbs, 1),
            offsets=(0, 0),
            block_shape=(BLOCK_N, BLOCK_DMODEL),
            order=(1, 0),
        )
    for start_n in range(0, cur_block_m_end, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        mask_n = (start_n + offs_n) < cur_block_m_end

        if not use_block_ptr:
            # load k in transposed way
            offs_k = ((cur_seq_extend_start_contiguous + start_n + offs_n[None, :]) * stride_kbs +
                      cur_kv_head * stride_kh + offs_d[:, None])
            k = tl.load(K_Extend + offs_k, mask=(mask_n[None, :]) & (mask_d[:, None]), other=0.0)
        else:
            # k:[BLOCK_N, 1, BLOCK_DMODEL]
            k = tl.load(k_block_ptr, boundary_check=(0, 1))
            # k = k.trans()
            k_block_ptr = tl.advance(k_block_ptr, (0, BLOCK_N))

        qk = tl.dot(q, k, out_dtype=tl.float32)
        if BLOCK_DPE > 0:
            offs_kpe = ((cur_seq_extend_start_contiguous + start_n + offs_n[None, :]) * stride_kbs +
                        cur_kv_head * stride_kh + offs_dpe[:, None])
            kpe = tl.load(
                K_Extend + offs_kpe,
                mask=mask_n[None, :],
                other=0.0,
            )
            qk += tl.dot(qpe, kpe)

        qk *= sm_scale

        if logit_cap > 0:
            qk = logit_cap * tanh(qk / logit_cap)

        mask_causual = (cur_block_m * BLOCK_M + offs_m[:, None]) >= (start_n + offs_n[None, :])
        mask_causual &= mask_m[:, None] & mask_n[None, :]
        qk = tl.where(mask_causual, qk, float('-inf'))

        n_e_max = tl.maximum(tl.max(qk, 1), e_max)
        re_scale = tl.exp(e_max - n_e_max)
        p = tl.exp(qk - n_e_max[:, None])
        deno = deno * re_scale + tl.sum(p, 1)
        if not use_block_ptr:
            offs_v = ((cur_seq_extend_start_contiguous + start_n + offs_n[:, None]) * stride_vbs +
                      cur_kv_head * stride_vh + offs_dv[None, :])
            v = tl.load(V_Extend + offs_v, mask=mask_n[:, None] & mask_dv[None, :], other=0.0)
        else:
            v = tl.load(v_block_ptr, boundary_check=(0, 1))
            v_block_ptr = tl.advance(v_block_ptr, (BLOCK_N, 0))

        p = p.to(v.dtype)
        acc = acc * re_scale[:, None] + tl.dot(p, v)

        e_max = n_e_max

    if not use_block_ptr:
        offs_o = ((cur_seq_extend_start_contiguous + cur_block_m * BLOCK_M + offs_m[:, None]) * stride_obs +
                  cur_head * stride_oh + offs_dv[None, :])
        tl.store(
            O_Extend + offs_o,
            acc / deno[:, None],
            mask=mask_m[:, None] & mask_dv[None, :],
        )
    else:
        # Ragged inputs: out:[S, H, D]
        offs_o = cur_seq_extend_start_contiguous * stride_obs + cur_head * stride_oh
        # the offset to the current seq
        o_block_ptr = tl.make_block_ptr(
            base=O_Extend + offs_o,
            shape=(cur_seq_len_extend, Lq),
            strides=(stride_obs, 1),
            offsets=(cur_block_m * BLOCK_M, 0),
            block_shape=(BLOCK_M, BLOCK_DMODEL),
            order=(1, 0),
        )
        # :[BLOCK_M, BLOCK_DMODEL]
        o = acc / deno[:, None]
        o = o.to(q.dtype)
        # o = o.reshape(BLOCK_M, 1, BLOCK_DMODEL)
        tl.store(o_block_ptr, o, boundary_check=(0, 1))


# pylint: disable=unused-argument
def extend_attention_fwd(
    q_extend,
    k_extend,
    v_extend,
    o_extend,
    k_buffer,
    v_buffer,
    req_to_tokens,
    b_req_idx,
    b_seq_len,
    b_seq_len_extend,
    b_start_loc_extend,
    max_len_extend,
    sm_scale=None,
    logit_cap=0.0,
):
    '''
    q_extend, k_extend, v_extend, o_extend: contiguous tensors

    k_buffer, v_buffer: (prefix + extend) tensors in mem_manager
    '''
    # print('=============== extend_attention_fwd ==============')
    # print(f'q_extend shape: {q_extend.shape}')
    # print(f'k_extend shape: {k_extend.shape}')
    # print(f'v_extend shape: {v_extend.shape}')
    # print(f'o_extend shape: {o_extend.shape}')
    # print(f'k_buffer shape: {k_buffer.shape}')
    # print(f'v_buffer shape: {v_buffer.shape}')
    # print(f'req_to_tokens shape: {req_to_tokens.shape}')
    # print(f'b_req_idx shape: {b_req_idx.shape}, value: {b_req_idx}')
    # print(f'b_seq_len shape: {b_seq_len.shape}, value: {b_seq_len}')
    # print(
    #     f'b_seq_len_extend shape: {b_seq_len_extend.shape}, value: {b_seq_len_extend}')
    # print(
    #     f'b_start_loc_extend shape: {b_start_loc_extend.shape}, value: {b_start_loc_extend}')
    # print(f'max_len_extend: {max_len_extend}')
    Lq, Lk, Lv = (
        q_extend.shape[-1],
        k_extend.shape[-1],
        v_extend.shape[-1],
    )
    if Lq == 576:
        BLOCK_DMODEL = 512
        BLOCK_DPE = 64
    elif Lq == 288:
        BLOCK_DMODEL = 256
        BLOCK_DPE = 32
    else:
        BLOCK_DMODEL = triton.next_power_of_2(Lq)
        BLOCK_DPE = 0
    BLOCK_DV = triton.next_power_of_2(Lv)

    if is_cuda_available and CUDA_CAPABILITY[0] >= 9:
        if Lq <= 256:
            BLOCK_M, BLOCK_N = (128, 64)
        else:
            BLOCK_M, BLOCK_N = (32, 64)
    elif is_cuda_available and CUDA_CAPABILITY[0] >= 8:
        if Lq <= 128:
            BLOCK_M, BLOCK_N = (128, 128)
        elif Lq <= 256:
            BLOCK_M, BLOCK_N = (64, 64)
        else:
            BLOCK_M, BLOCK_N = (32, 64)
    else:
        BLOCK_M, BLOCK_N = (64, 64) if Lq <= 128 else (32, 32)

    if is_xpu_available:
        BLOCK_M, BLOCK_N = (128, 64)

    sm_scale = sm_scale or 1.0 / (Lq**0.5)
    batch_size, head_num = b_seq_len.shape[0], q_extend.shape[1]
    kv_group_num = q_extend.shape[1] // k_extend.shape[1]
    # q/k/v: [seq_len, head_num, d_model]
    #
    grid = (head_num, triton.cdiv(max_len_extend, BLOCK_M), batch_size)
    # grid = (batch_size, head_num, triton.cdiv(max_len_extend, BLOCK_M))
    num_warps = 4 if Lk <= 64 else 8
    if is_xpu_available:
        num_warps = 8 if Lq == 64 else 16
    num_stages = 1
    b_seq_len_extend = b_seq_len_extend.to(torch.int32)
    # print('BLOCK_M', BLOCK_M, 'BLOCK_N', BLOCK_N, 'BLOCK_DMODEL', BLOCK_DMODEL, 'BLOCK_DPE', BLOCK_DPE, 'BLOCK_DV', BLOCK_DV)
    # print('Lq', Lq, 'Lv', Lv, 'kv_group_num', kv_group_num, 'num_warps', num_warps, 'num_stages', num_stages)
    # print(q_extend.shape, k_extend.shape, v_extend.shape, o_extend.shape, k_buffer.shape, v_buffer.shape, req_to_tokens.shape)
    _fwd_kernel2[grid](
        q_extend, k_extend, v_extend, o_extend, k_buffer, v_buffer, req_to_tokens, b_req_idx, b_seq_len,
        b_start_loc_extend, b_seq_len_extend, sm_scale, q_extend.shape[1], q_extend.stride(0), q_extend.stride(1),
        k_extend.stride(0), k_extend.stride(1), v_extend.stride(0), v_extend.stride(1), o_extend.stride(0),
        o_extend.stride(1), k_buffer.stride(0), k_buffer.stride(0), v_buffer.stride(0), v_buffer.stride(0),
        req_to_tokens.stride(0), logit_cap=logit_cap, BLOCK_DMODEL=BLOCK_DMODEL, BLOCK_DPE=BLOCK_DPE, BLOCK_DV=BLOCK_DV,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, Lq=Lq, kv_group_num=kv_group_num, use_block_ptr=1, advanced_path=True, Lv=Lv,
        num_warps=num_warps, num_stages=num_stages, grf_mode='large')


# pylint: disable=unused-argument
def gen_args(B, N_CTX, H_Q, H_KV, D, dtype, device):

    # b_seq_len_prefix = torch.randint(
    #     1, N_CTX // 2, (B,), dtype=torch.int32, device=device
    # )
    # b_seq_len_extend = torch.randint(
    #     1, N_CTX // 2, (B,), dtype=torch.int32, device=device
    # )
    # b_seq_len = b_seq_len_prefix + b_seq_len_extend
    b_seq_len_prefix = torch.full((B, ), 222219, dtype=torch.int32, device=device)
    b_seq_len_extend = torch.full((B, ), 1024, dtype=torch.int32, device=device)
    b_seq_len = torch.full((B, ), 223243, dtype=torch.int32, device=device)
    max_len_in_batch = torch.max(b_seq_len, 0)[0].item()

    b_req_idx = torch.arange(B, dtype=torch.int32, device=device)
    req_to_tokens = torch.empty((B, max_len_in_batch), dtype=torch.int32, device=device)

    b_start_loc = torch.zeros((B, ), dtype=torch.int32, device=device)
    b_start_loc[1:] = torch.cumsum(b_seq_len[:-1], 0)
    b_start_loc_extend = torch.zeros((B, ), dtype=torch.int32, device=device)
    b_start_loc_extend[1:] = torch.cumsum(b_seq_len_extend[:-1], 0)
    for i in range(B):
        req_to_tokens[i, :b_seq_len[i]] = torch.arange(b_start_loc[i], b_start_loc[i] + b_seq_len[i])

    total_token_num = torch.sum(b_seq_len).item()
    extend_token_num = torch.sum(b_seq_len_extend).item()
    k_buffer = torch.empty((total_token_num, H_KV, D), dtype=dtype, device=device).normal_(mean=0.1, std=0.2)
    v_buffer = torch.empty((total_token_num, H_KV, D), dtype=dtype, device=device).normal_(mean=0.1, std=0.2)

    k_extend = torch.empty((extend_token_num, H_KV, D), dtype=dtype, device=device)
    v_extend = torch.empty((extend_token_num, H_KV, D), dtype=dtype, device=device)
    q_extend = torch.empty((extend_token_num, H_Q, D), dtype=dtype, device=device)
    for i in range(B):
        extend_start_in_buffer = b_start_loc[i] + b_seq_len_prefix[i]
        extend_end_in_buffer = b_start_loc[i] + b_seq_len[i]
        extend_start = b_start_loc_extend[i]
        extend_end = b_start_loc_extend[i] + b_seq_len_extend[i]
        k_extend[extend_start:extend_end] = k_buffer[extend_start_in_buffer:extend_end_in_buffer]
        v_extend[extend_start:extend_end] = v_buffer[extend_start_in_buffer:extend_end_in_buffer]
        q_extend[extend_start:extend_end] = torch.empty((b_seq_len_extend[i], H_Q, D), dtype=dtype,
                                                        device=device).normal_(mean=0.1, std=0.2)

    o_extend = torch.empty((extend_token_num, H_Q, D), dtype=dtype, device=device)
    b_seq_len_extend = b_seq_len - b_seq_len_prefix
    b_start_loc_extend = torch.zeros_like(b_seq_len)
    b_start_loc_extend[1:] = torch.cumsum(b_seq_len_extend[:-1], 0)
    max_len_extend = torch.max(b_seq_len_extend, 0)[0].item()

    params = []
    params.append((q_extend, k_extend, v_extend, o_extend))
    params.append((k_buffer, v_buffer))
    params.append((req_to_tokens, max_len_extend))
    params.append((b_req_idx, b_seq_len, b_seq_len_extend, b_start_loc_extend))
    return params


# pylint: disable=unused-argument
@benchmark_suit.perf_report(
    benchmark_suit.Benchmark(
        # argument names to use as an x-axis for the plot
        x_names=['B', 'N_CTX', 'H_Q', 'H_KV', 'D', 'MODE', 'VALIDATE'],
        x_vals=[  #
            [bs, 1024, 32, 8, 128, 'fwd', False] for bs in [1]
        ],
        line_arg='provider',
        # argument name whose value corresponds to a different line in the plot
        # possible values for `line_arg``
        line_vals=[
            'triton',
        ],
        # label name for the lines
        line_names=[
            'Triton',
        ],
        # line styles
        styles=[('green', '-'), ('green', '--'), ('blue', '-'), ('blue', '--')],
        ylabel=['GB/s', 'TFlops'],  # label name for the y-axis
        plot_name='extended-attn-performance',
        # name for the plot. Used also as a file name for saving the plot.
        args={},
    ))
def benchmark(B, N_CTX, H_Q, H_KV, D, MODE, VALIDATE, provider):
    torch.manual_seed(0)

    # dtype = torch.bfloat16
    dtype = torch.float16
    params = gen_args(B, N_CTX, H_Q, H_KV, D, dtype, 'xpu')
    q_extend, k_extend, v_extend, o_extend = params[0]
    k_buffer, v_buffer = params[1]
    req_to_tokens, max_len_extend = params[2]
    b_req_idx, b_seq_len, b_seq_len_extend, b_start_loc_extend = params[3]
    custom_mask = None
    mask_indptr = None

    quantiles = [0.5, 0.0, 1.0]
    if provider == 'triton':

        def triton_fn():
            extend_attention_fwd(q_extend, k_extend, v_extend, o_extend, k_buffer, v_buffer, req_to_tokens, b_req_idx,
                                 b_seq_len, b_seq_len_extend, b_start_loc_extend, max_len_extend)

        _, min_ms, max_ms, mean, cv = benchmark_suit.do_bench(triton_fn, n_warmup=10, n_repeat=10, quantiles=quantiles)
        print(f'mean: {mean:.2f} ms, max: {max_ms:.2f} ms, min: {min_ms:.2f} ms')

        with torch.profiler.profile(activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.XPU,
        ]) as p:
            print('[info] start running')
            triton_fn()
        print(p.key_averages().table(sort_by='self_xpu_time_total', row_limit=-1))

    else:
        raise NotImplementedError(f'Unsupported provider {provider}')

    N_CTX_TOTAL = k_buffer.shape[0]
    N_CTX_EXTEND = k_extend.shape[0]
    tflops = lambda ms: (H_Q + H_KV) * (N_CTX_EXTEND + N_CTX_TOTAL) * N_CTX_TOTAL * D * (1e-12) / (ms * 1e-3)
    gbps = lambda ms: 2 * (N_CTX_EXTEND * (H_Q + H_KV) + N_CTX_TOTAL * H_KV) * D * 2 * (1e-9) / (ms * 1e-3)

    return (gbps(mean), gbps(max_ms), gbps(min_ms)), (tflops(mean), tflops(max_ms), tflops(min_ms)), cv


if __name__ == '__main__':
    benchmark.run(show_plots=False, print_data=True)
