import torch
from torch.nn.utils.rnn import pad_sequence


def pad(tensors, padding_value=0, total_length=None):
    size = [len(tensors)] + [
        max(tensor.size(i) for tensor in tensors)
        for i in range(len(tensors[0].size()))
    ]
    if total_length is not None:
        assert total_length >= size[1]
        size[1] = total_length
    out_tensor = tensors[0].data.new(*size).fill_(padding_value)
    for i, tensor in enumerate(tensors):
        out_tensor[i][[slice(0, i) for i in tensor.size()]] = tensor
    return out_tensor


def tarjan(sequence):
    r"""
    Tarjan algorithm for finding Strongly Connected Components (SCCs) of a graph.
    Args:
        sequence (list):
            List of head indices.
    Yields:
        A list of indices that make up a SCC. All self-loops are ignored.
    Examples:
        >>> next(tarjan([2, 5, 0, 3, 1]))  # (1 -> 5 -> 2 -> 1) is a cycle
        [2, 5, 1]
    """

    sequence = [-1] + sequence
    # record the search order, i.e., the timestep
    dfn = [-1] * len(sequence)
    # record the the smallest timestep in a SCC
    low = [-1] * len(sequence)
    # push the visited into the stack
    stack, onstack = [], [False] * len(sequence)

    def connect(i, timestep):
        dfn[i] = low[i] = timestep[0]
        timestep[0] += 1
        stack.append(i)
        onstack[i] = True

        for j, head in enumerate(sequence):
            if head != i:
                continue
            if dfn[j] == -1:
                yield from connect(j, timestep)
                low[i] = min(low[i], low[j])
            elif onstack[j]:
                low[i] = min(low[i], dfn[j])

        # a SCC is completed
        if low[i] == dfn[i]:
            cycle = [stack.pop()]
            while cycle[-1] != i:
                onstack[cycle[-1]] = False
                cycle.append(stack.pop())
            onstack[i] = False
            # ignore the self-loop
            if len(cycle) > 1:
                yield cycle

    timestep = [0]
    for i in range(len(sequence)):
        if dfn[i] == -1:
            yield from connect(i, timestep)


def chuliu_edmonds(s):
    r"""
    ChuLiu/Edmonds algorithm for non-projective decoding.
    Some code is borrowed from `tdozat's implementation`_.
    Descriptions of notations and formulas can be found in
    `Non-projective Dependency Parsing using Spanning Tree Algorithms`_.
    Notes:
        The algorithm does not guarantee to parse a single-root tree.
    References:
        - Ryan McDonald, Fernando Pereira, Kiril Ribarov and Jan Hajic. 2005.
          `Non-projective Dependency Parsing using Spanning Tree Algorithms`_.
    Args:
        s (~torch.Tensor): ``[seq_len, seq_len]``.
            Scores of all dependent-head pairs.
    Returns:
        ~torch.Tensor:
            A tensor with shape ``[seq_len]`` for the resulting non-projective parse tree.
    .. _tdozat's implementation:
        https://github.com/tdozat/Parser-v3
    .. _Non-projective Dependency Parsing using Spanning Tree Algorithms:
        https://www.aclweb.org/anthology/H05-1066/
    """

    s[0, 1:] = float('-inf')
    # prevent self-loops
    s.diagonal()[1:].fill_(float('-inf'))
    # select heads with highest scores
    tree = s.argmax(-1)
    # return the cycle finded by tarjan algorithm lazily
    cycle = next(tarjan(tree.tolist()[1:]), None)
    # if the tree has no cycles, then it is a MST
    if not cycle:
        return tree
    # indices of cycle in the original tree
    cycle = torch.tensor(cycle)
    # indices of noncycle in the original tree
    noncycle = torch.ones(len(s)).index_fill_(0, cycle, 0)
    noncycle = torch.where(noncycle.gt(0))[0]

    def contract(s):
        # heads of cycle in original tree
        cycle_heads = tree[cycle]
        # scores of cycle in original tree
        s_cycle = s[cycle, cycle_heads]

        # calculate the scores of cycle's potential dependents
        # s(c->x) = max(s(x'->x)), x in noncycle and x' in cycle
        s_dep = s[noncycle][:, cycle]
        # find the best cycle head for each noncycle dependent
        deps = s_dep.argmax(1)
        # calculate the scores of cycle's potential heads
        # s(x->c) = max(s(x'->x) - s(a(x')->x') + s(cycle)), x in noncycle and x' in cycle
        #                                                    a(v) is the predecessor of v in cycle
        #                                                    s(cycle) = sum(s(a(v)->v))
        s_head = s[cycle][:, noncycle] - s_cycle.view(-1, 1) + s_cycle.sum()
        # find the best noncycle head for each cycle dependent
        heads = s_head.argmax(0)

        contracted = torch.cat((noncycle, torch.tensor([-1])))
        # calculate the scores of contracted graph
        s = s[contracted][:, contracted]
        # set the contracted graph scores of cycle's potential dependents
        s[:-1, -1] = s_dep[range(len(deps)), deps]
        # set the contracted graph scores of cycle's potential heads
        s[-1, :-1] = s_head[heads, range(len(heads))]

        return s, heads, deps

    # keep track of the endpoints of the edges into and out of cycle for reconstruction later
    s, heads, deps = contract(s)

    # y is the contracted tree
    y = chuliu_edmonds(s)
    # exclude head of cycle from y
    y, cycle_head = y[:-1], y[-1]

    # fix the subtree with no heads coming from the cycle
    # len(y) denotes heads coming from the cycle
    subtree = y < len(y)
    # add the nodes to the new tree
    tree[noncycle[subtree]] = noncycle[y[subtree]]
    # fix the subtree with heads coming from the cycle
    subtree = ~subtree
    # add the nodes to the tree
    tree[noncycle[subtree]] = cycle[deps[subtree]]
    # fix the root of the cycle
    cycle_root = heads[cycle_head]
    # break the cycle and add the root of the cycle to the tree
    tree[cycle[cycle_root]] = noncycle[cycle_head]

    return tree


def mst(scores, mask, multiroot=False):
    r"""
    MST algorithm for decoding non-pojective trees.
    This is a wrapper for ChuLiu/Edmonds algorithm.
    The algorithm first runs ChuLiu/Edmonds to parse a tree and then have a check of multi-roots,
    If ``multiroot=True`` and there indeed exist multi-roots, the algorithm seeks to find
    best single-root trees by iterating all possible single-root trees parsed by ChuLiu/Edmonds.
    Otherwise the resulting trees are directly taken as the final outputs.
    Args:
        scores (~torch.Tensor): ``[batch_size, seq_len, seq_len]``.
            Scores of all dependent-head pairs.
        mask (~torch.BoolTensor): ``[batch_size, seq_len]``.
            The mask to avoid parsing over padding tokens.
            The first column serving as pseudo words for roots should be ``False``.
        muliroot (bool):
            Ensures to parse a single-root tree If ``False``.
    Returns:
        ~torch.Tensor:
            A tensor with shape ``[batch_size, seq_len]`` for the resulting non-projective parse trees.
    Examples:
        >>> scores = torch.tensor([[[-11.9436, -13.1464,  -6.4789, -13.8917],
                                    [-60.6957, -60.2866, -48.6457, -63.8125],
                                    [-38.1747, -49.9296, -45.2733, -49.5571],
                                    [-19.7504, -23.9066,  -9.9139, -16.2088]]])
        >>> scores[:, 0, 1:] = float('-inf')
        >>> scores.diagonal(0, 1, 2)[1:].fill_(float('-inf'))
        >>> mask = torch.tensor([[False,  True,  True,  True]])
        >>> mst(scores, mask)
        tensor([[0, 2, 0, 2]])
    """

    batch_size, seq_len, _ = scores.shape
    scores = scores.cpu().unbind()

    preds = []
    for i, length in enumerate(mask.sum(1).tolist()):
        s = scores[i][:length + 1, :length + 1]
        tree = chuliu_edmonds(s)
        roots = torch.where(tree[1:].eq(0))[0] + 1
        if not multiroot and len(roots) > 1:
            s_root = s[:, 0]
            s_best = float('-inf')
            s = s.index_fill(1, torch.tensor(0), float('-inf'))
            for root in roots:
                s[:, 0] = float('-inf')
                s[root, 0] = s_root[root]
                t = chuliu_edmonds(s)
                s_tree = s[1:].gather(1, t[1:].unsqueeze(-1)).sum()
                if s_tree > s_best:
                    s_best, tree = s_tree, t
        preds.append(tree)

    return pad(preds, total_length=seq_len).to(mask.device)


def stripe(x, n, w, offset=(0, 0), dim=1):
    r'''Returns a diagonal stripe of the tensor.

    Parameters:
        x (Tensor): the input tensor with 2 or more dims.
        n (int): the length of the stripe.
        w (int): the width of the stripe.
        offset (tuple): the offset of the first two dims.
        dim (int): 0 if returns a horizontal stripe; 1 else.

    Example::
    >>> x = torch.arange(25).view(5, 5)
    >>> x
    tensor([[ 0,  1,  2,  3,  4],
            [ 5,  6,  7,  8,  9],
            [10, 11, 12, 13, 14],
            [15, 16, 17, 18, 19],
            [20, 21, 22, 23, 24]])
    >>> stripe(x, 2, 3, (1, 1))
    tensor([[ 6,  7,  8],
            [12, 13, 14]])
    >>> stripe(x, 2, 3, dim=0)
    tensor([[ 0,  5, 10],
            [ 6, 11, 16]])
    '''
    x, seq_len = x.contiguous(), x.size(1)
    stride, numel = list(x.stride()), x[0, 0].numel()
    stride[0] = (seq_len + 1) * numel
    stride[1] = (1 if dim == 1 else seq_len) * numel
    return x.as_strided(size=(n, w, *x.shape[2:]),
                        stride=stride,
                        storage_offset=(offset[0] * seq_len + offset[1]) *
                        numel)


def eisner(scores, mask):
    lens = mask.sum(1)
    batch_size, seq_len, _ = scores.shape
    scores = scores.permute(2, 1, 0)
    s_i = torch.full_like(scores, float('-inf'))
    s_c = torch.full_like(scores, float('-inf'))
    p_i = scores.new_zeros(seq_len, seq_len, batch_size).long()
    p_c = scores.new_zeros(seq_len, seq_len, batch_size).long()
    s_c.diagonal().fill_(0)

    for w in range(1, seq_len):  # span width
        n = seq_len - w  # span max end index
        starts = p_i.new_tensor(range(n)).unsqueeze(0)  # span start index
        # ilr = C(i->r) + C(j->r+1)
        ilr = stripe(s_c, n, w) + stripe(s_c, n, w, (w, 1))
        # [batch_size, n, w]
        ilr = ilr.permute(2, 0, 1)
        il = ilr + scores.diagonal(-w).unsqueeze(-1)
        # I(j->i) = max(C(i->r) + C(j->r+1) + s(j->i)), i <= r < j
        il_span, il_path = il.max(-1)
        s_i.diagonal(-w).copy_(il_span)
        p_i.diagonal(-w).copy_(il_path + starts)
        ir = ilr + scores.diagonal(w).unsqueeze(-1)
        # I(i->j) = max(C(i->r) + C(j->r+1) + s(i->j)), i <= r < j
        ir_span, ir_path = ir.max(-1)
        s_i.diagonal(w).copy_(ir_span)
        p_i.diagonal(w).copy_(ir_path + starts)

        # C(j->i) = max(C(r->i) + I(j->r)), i <= r < j
        cl = stripe(s_c, n, w, (0, 0), 0) + stripe(s_i, n, w, (w, 0))
        cl_span, cl_path = cl.permute(2, 0, 1).max(-1)
        s_c.diagonal(-w).copy_(cl_span)
        p_c.diagonal(-w).copy_(cl_path + starts)
        # C(i->j) = max(I(i->r) + C(r->j)), i < r <= j
        cr = stripe(s_i, n, w, (0, 1)) + stripe(s_c, n, w, (1, w), 0)
        cr_span, cr_path = cr.permute(2, 0, 1).max(-1)
        s_c.diagonal(w).copy_(cr_span)
        s_c[0, w][lens.ne(w)] = float('-inf')
        p_c.diagonal(w).copy_(cr_path + starts + 1)

    def backtrack(p_i, p_c, heads, i, j, complete):
        if i == j:
            return
        if complete:
            r = p_c[i, j]
            backtrack(p_i, p_c, heads, i, r, False)
            backtrack(p_i, p_c, heads, r, j, True)
        else:
            r, heads[j] = p_i[i, j], i
            i, j = sorted((i, j))
            backtrack(p_i, p_c, heads, i, r, True)
            backtrack(p_i, p_c, heads, j, r + 1, True)

    predicts = []
    p_c = p_c.permute(2, 0, 1).cpu()
    p_i = p_i.permute(2, 0, 1).cpu()
    for i, length in enumerate(lens.tolist()):
        heads = p_c.new_ones(length + 1, dtype=torch.long)
        backtrack(p_i[i], p_c[i], heads, 0, length, True)
        predicts.append(heads.to(mask.device))

    return pad_sequence(predicts, True)


# import numpy as np

# def parse_proj(scores, gold=None):
#     '''
#     Parse using Eisner's algorithm.
#     '''
#     nr, nc = np.shape(scores)
#     if nr != nc:
#         raise ValueError("scores must be a squared matrix with nw+1 rows")

#     N = nr - 1  # Number of words (excluding root).

#     # Initialize CKY table.
#     complete = np.zeros([N + 1, N + 1, 2])  # s, t, direction (right=1).
#     incomplete = np.zeros([N + 1, N + 1, 2])  # s, t, direction (right=1).
#     complete_backtrack = -np.ones([N + 1, N + 1, 2],
#                                   dtype=int)  # s, t, direction (right=1).
#     incomplete_backtrack = -np.ones([N + 1, N + 1, 2],
#                                     dtype=int)  # s, t, direction (right=1).

#     incomplete[0, :, 0] -= np.inf

#     # Loop from smaller items to larger items.
#     for k in range(1, N + 1):
#         for s in range(N - k + 1):
#             t = s + k

#             # First, create incomplete items.
#             # left tree
#             incomplete_vals0 = complete[s, s:t, 1] + complete[(s + 1):(
#                 t + 1), t, 0] + scores[t, s] + (0.0 if gold is not None
#                                                 and gold[s] == t else 1.0)
#             incomplete[s, t, 0] = np.max(incomplete_vals0)
#             incomplete_backtrack[s, t, 0] = s + np.argmax(incomplete_vals0)
#             # right tree
#             incomplete_vals1 = complete[s, s:t, 1] + complete[(s + 1):(
#                 t + 1), t, 0] + scores[s, t] + (0.0 if gold is not None
#                                                 and gold[t] == s else 1.0)
#             incomplete[s, t, 1] = np.max(incomplete_vals1)
#             incomplete_backtrack[s, t, 1] = s + np.argmax(incomplete_vals1)

#             # Second, create complete items.
#             # left tree
#             complete_vals0 = complete[s, s:t, 0] + incomplete[s:t, t, 0]
#             complete[s, t, 0] = np.max(complete_vals0)
#             complete_backtrack[s, t, 0] = s + np.argmax(complete_vals0)
#             # right tree
#             complete_vals1 = incomplete[s, (s + 1):(t + 1), 1] + complete[
#                 (s + 1):(t + 1), t, 1]
#             complete[s, t, 1] = np.max(complete_vals1)
#             complete_backtrack[s, t, 1] = s + 1 + np.argmax(complete_vals1)

#     value = complete[0][N][1]
#     heads = [-1 for _ in range(N + 1)]  # -np.ones(N+1, dtype=int)
#     backtrack_eisner(incomplete_backtrack, complete_backtrack, 0, N, 1, 1,
#                      heads)

#     value_proj = 0.0
#     for m in range(1, N + 1):
#         h = heads[m]
#         value_proj += scores[h, m]

#     return heads

# def backtrack_eisner(incomplete_backtrack, complete_backtrack, s, t, direction,
#                      complete, heads):
#     '''
#     Backtracking step in Eisner's algorithm.
#     - incomplete_backtrack is a (NW+1)-by-(NW+1) numpy array indexed by a start position,
#     an end position, and a direction flag (0 means left, 1 means right). This array contains
#     the arg-maxes of each step in the Eisner algorithm when building *incomplete* spans.
#     - complete_backtrack is a (NW+1)-by-(NW+1) numpy array indexed by a start position,
#     an end position, and a direction flag (0 means left, 1 means right). This array contains
#     the arg-maxes of each step in the Eisner algorithm when building *complete* spans.
#     - s is the current start of the span
#     - t is the current end of the span
#     - direction is 0 (left attachment) or 1 (right attachment)
#     - complete is 1 if the current span is complete, and 0 otherwise
#     - heads is a (NW+1)-sized numpy array of integers which is a placeholder for storing the
#     head of each word.
#     '''
#     if s == t:
#         return
#     if complete:
#         r = complete_backtrack[s][t][direction]
#         if direction == 0:
#             backtrack_eisner(incomplete_backtrack, complete_backtrack, s, r, 0,
#                              1, heads)
#             backtrack_eisner(incomplete_backtrack, complete_backtrack, r, t, 0,
#                              0, heads)
#             return
#         else:
#             backtrack_eisner(incomplete_backtrack, complete_backtrack, s, r, 1,
#                              0, heads)
#             backtrack_eisner(incomplete_backtrack, complete_backtrack, r, t, 1,
#                              1, heads)
#             return
#     else:
#         r = incomplete_backtrack[s][t][direction]
#         if direction == 0:
#             heads[s] = t
#             backtrack_eisner(incomplete_backtrack, complete_backtrack, s, r, 1,
#                              1, heads)
#             backtrack_eisner(incomplete_backtrack, complete_backtrack, r + 1,
#                              t, 0, 1, heads)
#             return
#         else:
#             heads[t] = s
#             backtrack_eisner(incomplete_backtrack, complete_backtrack, s, r, 1,
#                              1, heads)
#             backtrack_eisner(incomplete_backtrack, complete_backtrack, r + 1,
#                              t, 0, 1, heads)
#             return