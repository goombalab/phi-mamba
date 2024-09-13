import torch
import torch.nn.functional as F
from einops import rearrange, repeat


def segsum(x):
    """More stable segment sum calculation."""
    # [1, 2, 3]
    T = x.size(-1)
    x = repeat(x, "... d -> ... d e", e=T)
    # [[1, 1, 1], [2, 2, 2], [3, 3, 3]]
    mask = torch.tril(torch.ones(T, T, device=x.device, dtype=bool), diagonal=-1)
    x = x.masked_fill(~mask, 0)
    # [[0, 0, 0], [2, 0, 0], [3, 3, 0]]
    x_segsum = torch.cumsum(x, dim=-2)
    # [[0, 0, 0], [2, 0, 0], [5, 3, 0]]
    mask = torch.tril(torch.ones(T, T, device=x.device, dtype=bool), diagonal=0)
    x_segsum = x_segsum.masked_fill(~mask, -torch.inf)
    return x_segsum

def ssd_minimal_discrete(X, A_log, B, C, block_len, initial_states=None):
    """
    Arguments:
        X: (batch, length, n_heads, d_head)
        A_log: (batch, length, n_heads)
        B: (batch, length, n_heads, d_state)
        C: (batch, length, n_heads, d_state)
        block_len: int
        initial_states: (batch, n_heads, d_state, d_state) or None
    Return:
        Y: (batch, length, n_heads, d_head)
        final_state: (batch, n_heads, d_head, d_state)
    """
    assert X.dtype == A_log.dtype == B.dtype == C.dtype
    assert X.shape[1] % block_len == 0
    batch_size, length, n_heads, d_head = X.shape
    d_state = B.shape[-1]
    assert A_log.shape == (batch_size, length, n_heads)
    assert B.shape == C.shape == (batch_size, length, n_heads, d_state)

    # Rearrange into blocks/chunks
    X, A_log, B, C = [
        rearrange(x, "b (c l) ... -> b c l ...", l=block_len) for x in (X, A_log, B, C)
    ]

    A_log = rearrange(A_log, "b c l h -> b h c l")
    A_cumsum = torch.cumsum(A_log, dim=-1)

    # 1. Compute the output for each intra-chunk (diagonal blocks)
    length = torch.exp(segsum(A_log))
    Y_diag = torch.einsum("bclhn,bcshn,bhcls,bcshp->bclhp", C, B, length, X)

    # 2. Compute the state for each intra-chunk
    # (right term of low-rank factorization of off-diagonal blocks; B terms)
    decay_states = torch.exp((A_cumsum[:, :, :, -1:] - A_cumsum))
    states = torch.einsum("bclhn,bhcl,bclhp->bchpn", B, decay_states, X)

    # 3. Compute the inter-chunk SSM recurrence; produces correct SSM states at chunk boundaries
    # (middle term of factorization of off-diag blocks; A terms)
    if initial_states is None:
        initial_states = torch.zeros_like(states[:, :1])
    else:
        # Rearrange initial states into blocks
        initial_states = rearrange(initial_states, "b h d s -> b 1 h d s")

    states = torch.cat([initial_states, states], dim=1)
    decay_chunk = torch.exp(segsum(F.pad(A_cumsum[:, :, :, -1], (1, 0))))
    new_states = torch.einsum("bhzc,bchpn->bzhpn", decay_chunk, states)
    states, final_state = new_states[:, :-1], new_states[:, -1]

    # 4. Compute state -> output conversion per chunk
    # (left term of low-rank factorization of off-diagonal blocks; C terms)
    state_decay_out = torch.exp(A_cumsum)
    Y_off = torch.einsum("bclhn,bchpn,bhcl->bclhp", C, states, state_decay_out)

    # Add output of intra-chunk and inter-chunk terms (diagonal and off-diagonal blocks)
    Y = rearrange(Y_diag + Y_off, "b c l h p -> b (c l) h p")
    return Y, final_state

def step(x, B, C, A_log, state):
    """
    Arguments:
        x: (batch, n_v_heads, dim)
        B: (batch, n_qk_heads, d_state)
        C: (batch, n_qk_heads, d_state)
        A_log: (batch, length, n_heads)
        state: dict
    Return:
        y: (batch, n_v_heads, dim)
        ssm_state: (batch, n_v_heads, d_state)
    """
    n_v_heads = x.shape[1]
    n_qk_heads = B.shape[1]
    assert n_v_heads % n_qk_heads == 0
    

    # Broadcast B and C across the head dimension (n_v_heads % n_qk_heads == 0)
    B = B.repeat_interleave(n_v_heads // n_qk_heads, dim=1)
    C = C.repeat_interleave(n_v_heads // n_qk_heads, dim=1)

    # SSM step
    Bx = torch.einsum("bhn,bhp->bhpn", B, x)
    Ah = torch.einsum(
        "bh,bhpn->bhpn",
        torch.sigmoid(-A_log).to(x.dtype),  # = torch.exp(-F.softplus(A_log))
        state["ssm"].to(x.dtype),
    )
    ssm_state = Ah + Bx
    y = torch.einsum("bhn,bhpn->bhp", C, ssm_state)
    return y, ssm_state

def materialize_mixer(A_log, B, C, D):
    """
    Since the transfer matrix will be equated to the attention matrix,
    we need to support the form: torch.matmul(attn_weights, value_states).
    Thus, y = torch.matmul(T, X)
    Arguments:
        A_log: (batch, length, n_heads)
        B: (batch, length, n_heads, d_state)
        C: (batch, length, n_heads, d_state)
    Return:
        T: (batch, n_heads, length, length)
    """
    batch_size, length, n_heads, d_state = B.shape
    assert A_log.shape == (batch_size, length, n_heads)
    assert B.shape == C.shape == (batch_size, length, n_heads, d_state)

    # Compute:
    A_log = rearrange(-F.softplus(A_log), "b l h -> b h l")
    powers = torch.exp(segsum(A_log))
    T = torch.einsum("blhn,bshn,bhls->bhsl", C, B, powers)

    # Add D:
    if D is not None:
        T[:, :, torch.arange(length), torch.arange(length)] += D.view(1, n_heads, 1)

    T = rearrange(T, "b h z l -> b h l z")
    return T
