import torch
import torch.linalg as LA
import numpy as np
def Group(A, shapeA):
    """ transpose + reshape """
    dimA = torch.tensor(A.shape)
    rankA = len(shapeA)

    shapeB = []
    for i in range(0, rankA):
        shapeB += [torch.prod(dimA[shapeA[i]])]

    orderB = sum(shapeA, [])
    A = torch.reshape(torch.permute(A, orderB), shapeB)
    return A

def NCon(Tensor, Index):
    ConList = list(range(1, max(sum(Index, [])) + 1))

    while len(ConList) > 0:
        Icon = []
        for i in range(len(Index)):
            if ConList[0] in Index[i]:
                Icon.append(i)
                if len(Icon) == 2:
                    break

        if len(Icon) == 1:
            IndCommon = list(
                set([x for x in Index[Icon[0]] if Index[Icon[0]].count(x) > 1])
            )

            for icom in range(len(IndCommon)):
                Pos = sorted(
                    [i for i, x in enumerate(Index[Icon[0]]) if x == IndCommon[icom]]
                )
                Tensor[Icon[0]] = torch.diagonal(
                    Tensor[Icon[0]], dim1=Pos[0], dim2=Pos[1]
                ).sum(dim=-1)
                Index[Icon[0]].pop(Pos[1])
                Index[Icon[0]].pop(Pos[0])

        else:
            IndCommon = list(set(Index[Icon[0]]) & set(Index[Icon[1]]))
            Pos = [[], []]
            for i in range(2):
                for ind in range(len(IndCommon)):
                    Pos[i].append(Index[Icon[i]].index(IndCommon[ind]))
            A = torch.tensordot(Tensor[Icon[0]], Tensor[Icon[1]], dims=(Pos[0], Pos[1]))

            for i in range(2):
                for ind in range(len(IndCommon)):
                    Index[Icon[i]].remove(IndCommon[ind])
            Index[Icon[0]] = Index[Icon[0]] + Index[Icon[1]]
            Index.pop(Icon[1])
            Tensor[Icon[0]] = A
            Tensor.pop(Icon[1])

        ConList = list(set(ConList) ^ set(IndCommon))

    while len(Index) > 1:
        Tensor[0] = torch.outer(Tensor[0].flatten(), Tensor[1].flatten()).reshape(
            *Tensor[0].shape, *Tensor[1].shape
        )
        Tensor.pop(1)
        Index[0] = Index[0] + Index[1]
        Index.pop(1)

    Index = Index[0]
    if len(Index) > 0:
        Order = sorted(range(len(Index)), key=lambda k: Index[k])[::-1]
        Tensor = Tensor[0].permute(Order)
    else:
        Tensor = Tensor[0]

    return Tensor


class ComplexSVD(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        U, S, Vh = torch.linalg.svd(input, full_matrices=False)
        ctx.save_for_backward(input, U, S, Vh)
        return U, S, Vh

    @staticmethod
    def backward(ctx, grad_U, grad_S, grad_Vh):
        input, U, S, Vh = ctx.saved_tensors
        m, n = input.shape
        k = S.shape[-1]

        # Make S matrix
        S_mat = torch.diag_embed(S.type(torch.cfloat))
        S_inv = torch.where(torch.abs(S_mat) > 1e-12, 1.0 / S_mat, torch.zeros_like(S_mat))

        # Construct outer differences matrix
        F = S.unsqueeze(-2)**2 - S.unsqueeze(-1)**2
        F_inv = torch.where(torch.abs(F) > 1e-12, 1.0 / F, torch.zeros_like(F))

        # Project grad_U and grad_Vh
        Ut_dU = U.T.conj() @ grad_U
        Vt_dV = Vh @ grad_Vh.T.conj()

        Su = F_inv * (Ut_dU - Ut_dU.T.conj())
        Sv = F_inv * (Vt_dV - Vt_dV.T.conj())
        L = torch.eye(k, dtype=torch.cfloat) * (Vt_dV.T.conj() - Vt_dV)

        dA = U @ (torch.diag_embed(grad_S) + Su @ S_mat + S_mat @ Sv + 1/2 * S_inv @ L) @ Vh + (torch.eye(m) - U @ U.T.conj()) @ grad_U @ S_inv @ Vh + U @ S_inv @ grad_Vh @ (torch.eye(n) - Vh.T.conj() @ Vh)

        return dA


def complex_svd(x):
    return ComplexSVD.apply(x)
    

def Mps_LQP(T, UR, icheck=0):
    """ (0-T-L)(0-UR-1) -> (0-UL-1)(0-Tnew-L) using PyTorch """
    shapeT = torch.tensor(T.shape)
    rankT = len(shapeT)

    A = torch.tensordot(T, UR, dims=([rankT-1], [0]))
    shapeT = A.shape
    A = A.reshape(shapeT[0], -1)
    Tnew, UL = LA.qr(torch.transpose(A, 0, 1), mode='reduced')
    UL = torch.transpose(UL, 0, 1)
    Tnew = torch.transpose(Tnew, 0, 1)
    Tnew = Tnew.reshape(-1, *shapeT[1:])

    if icheck == 1:
        A = A.reshape(shapeT)
        B = torch.tensordot(UL, Tnew, dims=([1], [0]))
        print(torch.norm(A-B) / torch.norm(A))

    return UL, Tnew


def Mps_QRP(UL, T, icheck=0):
    """ (0-UL-1)(0-T-L) -> (0-Tnew-L)(0-UR-1) using PyTorch """
    shapeT = torch.tensor(T.shape)
    rankT = len(shapeT)

    A = torch.tensordot(UL, T, dims=([1], [0]))
    shapeT = A.shape
    A = A.reshape(-1, shapeT[-1])
    Tnew, UR = LA.qr(A, mode='reduced')
    Tnew = Tnew.reshape(*shapeT[:-1], -1)

    if icheck == 1:
        A = A.reshape(shapeT)
        B = torch.tensordot(Tnew, UR, dims=([rankT-1], [0]))
        print(torch.norm(A-B) / torch.norm(A))

    return Tnew, UR


def MPS_can_torch(T):
    Ns = len(T)
    T_can = [None]*Ns
    U = torch.eye(1, dtype=torch.cfloat)
    for i in range(Ns-1, 0, -1): # Right-canonical
        U, T_can[i] = Mps_LQP(T[i], U)
    T_can[0] = NCon([T[0], U], [[-1, -2, 1], [1, -3]])
    return T_can


def MPS_trun_Ds_torch(T, Ds): # Calculate the truncation
    Ns = len(T)
    T_trun_1 = [None]*Ns
    T_trun_2 = [None]*Ns
    T_trun_1[0] = 1 * T[0]
    for i in range(Ns - 1):
        A = NCon([T_trun_1[i], T[i+1]], [[-1, -2, 1], [1, -3, -4]])
        Da = A.shape
        A = Group(A, [[0, 1], [2, 3]])
        U, S, V = complex_svd(A)
        Dc = min(len(S), Ds)

        T_trun_2[i] = torch.reshape(U[:, :Dc], (Da[0], Da[1], -1))
        T_trun_1[i+1] = torch.reshape(torch.matmul(torch.diag(S[:Dc].type(torch.cfloat)), V[:Dc]), (-1, Da[2], Da[3]))
    
    T_trun_2[-1] = T_trun_1[-1] / torch.sqrt(NCon([T_trun_1[-1], torch.conj(T_trun_1[-1])], [[1, 2, 3], [1, 2, 3]]))
    return T_trun_2


def MPO_can_torch(T, Dp):
    Ns = len(T)
    T_can = [None]*Ns
    U = torch.eye(1, dtype=torch.cfloat)
    for i in range(Ns-1, 0, -1): # Right-canonical
        Dt = T[i].shape
        U, Q = Mps_LQP(Group(T[i], [[0], [1, 3], [2]]), U)
        T_can[i] = Q.reshape(U.shape[1], Dt[1], Dt[3], -1).permute(0, 1, 3, 2) * np.sqrt(Dp)
    Q = NCon([T[0], U], [[-1, -2, 1, -4], [1, -3]])
    T_can[0] = Q / torch.sqrt(NCon([Q, Q.conj()], [[1, 2, 3, 4], [1, 2, 3, 4]])) * np.sqrt(Dp)
    return T_can


def MPO_trun_Ds_torch(T, Ds, Dp): # Calculate the truncation
    Ns = len(T)
    T_trun_1 = [None]*Ns
    T_trun_2 = [None]*Ns
    T_trun_1[0] = 1 * T[0]
    for i in range(Ns - 1):
        A = NCon([T_trun_1[i], T[i+1]], [[-1, -2, 1, -3], [1, -4, -5, -6]])
        Da = A.shape
        A = Group(A, [[0, 1, 2], [3, 4, 5]])
        U, S, V = complex_svd(A)
        Dc = min(len(S), Ds)

        T_trun_2[i] = torch.reshape(U[:, :Dc], (Da[0], Da[1], Da[2], Dc)).permute(0, 1, 3, 2) * np.sqrt(Dp)
        T_trun_1[i+1] = torch.reshape(torch.matmul(torch.diag(S[:Dc].type(torch.cfloat)), V[:Dc]), (Dc, Da[3], Da[4], Da[5]))
    
    T_trun_2[-1] = T_trun_1[-1] / torch.sqrt(NCon([T_trun_1[-1], torch.conj(T_trun_1[-1])], [[1, 2, 3, 4], [1, 2, 3, 4]])) * np.sqrt(Dp)
    # print(Overlap_MPO(T, T_trun_2)/Dp**Ns)
    return T_trun_2


def qr_haar(Dp, n):
    """Generate a Haar-random matrix using the QR decomposition."""
    # Step 1
    N = Dp ** n
    # np.random.seed(0)
    Z = torch.randn(N, N, dtype=torch.cfloat)

    # Step 2
    Q, R = LA.qr(Z)
    # Step 3
    Lambda = torch.diag(torch.sgn(torch.diagonal(R)))

    # Step 4
    return torch.tensordot(Q, Lambda, dims=([1], [0]))


def Generate_Gate(Ns, Dp, depth, train=False):
    gate_all = []
    for d in range(depth):
        for i in range(Ns):
            gate = qr_haar(Dp, 1)
            if train == True:
                gate = gate.clone().detach().requires_grad_(True)
            gate_all.append(gate)
    return gate_all


def CR_Gate(Dp, k):
    I = torch.tensor([[1, 0], [0, 1]], dtype=torch.cfloat)
    Rk = torch.tensor([[1, 0], [0, np.exp(2j*np.pi/(2**k))]], dtype=torch.cfloat)
    
    CR = [None]*k
    CR[0] = torch.zeros(1, Dp, 2, Dp, dtype=torch.cfloat)
    CR[0][0, :, 0, :] = I
    CR[0][0, :, 1, :] = Rk
    for i in range(1, k-1):
        CR[i] = torch.zeros(2, Dp, 2, Dp, dtype=torch.cfloat)
        CR[i][0, :, 0, :] = I
        CR[i][1, :, 1, :] = I
    CR[-1] = torch.zeros(2, Dp, 1, Dp, dtype=torch.cfloat)
    CR[-1][0, :, 0, :] = torch.tensor([[1, 0], [0, 0]], dtype=torch.cfloat)
    CR[-1][1, :, 0, :] = torch.tensor([[0, 0], [0, 1]], dtype=torch.cfloat)
    return CR


def MPS_Circuit(gate_all, Ns, Ds, Dp, depth, trun=False):
    MPS = [torch.tensor([1, 0], dtype=torch.cfloat).reshape(1, Dp, 1) for i in range(Ns)]
    num = 0
    for d in range(depth):
        for i in range(Ns):
            gate = gate_all[num]
            num += 1
            MPS[i] = torch.einsum('ijk,mj->imk', MPS[i], gate)
        for i in range(0, Ns-1, 2):
            CNOT0 = torch.tensor([[[1, 0], [0, 0]], [[0, 0], [0, 1]]], dtype=torch.cfloat)
            CNOT1 = torch.tensor([[[1, 0], [0, 1]], [[0, 1], [1, 0]]], dtype=torch.cfloat)
            MPS[i] = Group(torch.einsum('ijk,nmj->imkn', MPS[i], CNOT0), [[0], [1], [2, 3]])
            MPS[i+1] = Group(torch.einsum('ijk,nmj->imkn', MPS[i+1], CNOT1), [[0, 3], [1], [2]])

        for i in range(Ns):
            gate = gate_all[num]
            num += 1
            MPS[i] = torch.einsum('ijk,mj->imk', MPS[i], gate)

        for i in range(1, Ns-1, 2):
            CNOT0 = torch.tensor([[[1, 0], [0, 0]], [[0, 0], [0, 1]]], dtype=torch.cfloat)
            CNOT1 = torch.tensor([[[1, 0], [0, 1]], [[0, 1], [1, 0]]], dtype=torch.cfloat)
            MPS[i] = Group(torch.einsum('ijk,nmj->imkn', MPS[i], CNOT0), [[0], [1], [2, 3]])
            MPS[i+1] = Group(torch.einsum('ijk,nmj->imkn', MPS[i+1], CNOT1), [[0, 3], [1], [2]])

        MPS = MPS_can_torch(MPS)
        if trun == True:
            MPS = MPS_trun_Ds_torch(MPS, Ds)
    else:
        return MPS


def MPO_Circuit(gate_all, Ns, Ds, Dp, depth, trun=False):
    MPO = [torch.eye(Dp, dtype=torch.cfloat).reshape(1, Dp, 1, Dp) for i in range(Ns)]
    num = 0
    for d in range(depth):
        for i in range(Ns):
            gate = gate_all[num]
            num += 1
            MPO[i] = torch.einsum('ijkl,mj->imkl', MPO[i], gate)
        for i in range(0, Ns-1, 2):
            CNOT0 = torch.tensor([[[1, 0], [0, 0]], [[0, 0], [0, 1]]], dtype=torch.cfloat)
            CNOT1 = torch.tensor([[[1, 0], [0, 1]], [[0, 1], [1, 0]]], dtype=torch.cfloat)
            MPO[i] = Group(torch.einsum('ijkl,nmj->imkln', MPO[i], CNOT0), [[0], [1], [2, 4], [3]])
            MPO[i+1] = Group(torch.einsum('ijkl,nmj->imkln', MPO[i+1], CNOT1), [[0, 4], [1], [2], [3]])

        for i in range(Ns):
            gate = gate_all[num]
            num += 1
            MPO[i] = torch.einsum('ijkl,mj->imkl', MPO[i], gate)

        for i in range(1, Ns-1, 2):
            CNOT0 = torch.tensor([[[1, 0], [0, 0]], [[0, 0], [0, 1]]], dtype=torch.cfloat)
            CNOT1 = torch.tensor([[[1, 0], [0, 1]], [[0, 1], [1, 0]]], dtype=torch.cfloat)
            MPO[i] = Group(torch.einsum('ijkl,nmj->imkln', MPO[i], CNOT0), [[0], [1], [2, 4], [3]])
            MPO[i+1] = Group(torch.einsum('ijkl,nmj->imkln', MPO[i+1], CNOT1), [[0, 4], [1], [2], [3]])

        MPO = MPO_can_torch(MPO)
        if trun == True:
            MPO = MPO_trun_Ds_torch(MPO, Ds)
    else:
        return MPO
    

def MPS_Circuit_QFT(Ns, Ds, Dp):
    MPS = [torch.tensor([1, -1], dtype=torch.cfloat).reshape(1, Dp, 1) / np.sqrt(2) for i in range(Ns)]
    H = torch.tensor([[1, 1], [1, -1]], dtype=torch.cfloat) / np.sqrt(2)
    for d in range(Ns):
        MPS[d] = torch.einsum('ijk,mj->imk', MPS[d], H)
        for k in range(2, Ns+1-d):
            Gate = CR_Gate(Dp, k)
            for i in range(d, d+k):
                MPS[i] = Group(torch.einsum('ijk,lmnj->ilmkn', MPS[i], Gate[i-d]), [[0, 1], [2], [3, 4]])
            MPS = MPS_can_torch(MPS)
            MPS = MPS_trun_Ds_torch(MPS, Ds)
    return MPS


def MPO_Circuit_QFT(Ns, Ds, Dp):
    MPO = [torch.eye(Dp, dtype=torch.cfloat).reshape(1, Dp, 1, Dp) for i in range(Ns)]
    MPO_all = []
    EE_all = []
    H = torch.tensor([[1, 1], [1, -1]], dtype=torch.cfloat) / np.sqrt(2)
    for d in range(Ns):
        MPO[d] = torch.einsum('ijkl,mj->imkl', MPO[d], H)
        for k in range(2, Ns+1-d):
            Gate = CR_Gate(Dp, k)
            for i in range(d, d+k):
                MPO[i] = Group(torch.einsum('ijko,lmnj->ilmkno', MPO[i], Gate[i-d]), [[0, 1], [2], [3, 4], [5]])
            MPO = MPO_can_torch(MPO)
            MPO = MPO_trun_Ds_torch(MPO, Ds)
    return MPO


def Overlap_MPO(MPO1, MPO2):
    Ns = len(MPO1)
    TL = torch.eye(1, dtype=torch.cfloat)
    for i in range(Ns):
        TL = torch.einsum('ij,iklm,jknm->ln', TL, MPO1[i], torch.conj(MPO2[i]))
    return torch.trace(TL).abs()


def Overlap_MPS(MPS1, MPS2):
    Ns = len(MPS1)
    TL = torch.eye(1, dtype=torch.cfloat)
    for i in range(Ns):
        TL = torch.einsum('ij,ikl,jkm->lm', TL, MPS1[i], torch.conj(MPS2[i]))
    return torch.trace(TL).abs()


def Cal_Obs_MPS(MPS, Obs_list, pos_list):
    Ns = len(MPS)
    MPS_copy = MPS.copy()
    for i in range(len(pos_list)):
        MPS[pos_list[i]] = torch.einsum('ijk,lj->ilk', MPS[pos_list[i]], Obs_list[i])
    Ns = len(MPS)
    TL = torch.eye(1, dtype=torch.cfloat)
    for i in range(Ns):
        TL = torch.einsum('ij,ikl,jkm->lm', TL, MPS[i], torch.conj(MPS_copy[i]))
    return torch.trace(TL).real
