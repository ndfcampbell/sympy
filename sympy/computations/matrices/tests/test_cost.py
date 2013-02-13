from sympy.computations.matrices.cost import (flops, numelements,
        nbytesofoutputs, nbytesofinputs, commcost, memtime, compcost)
from sympy import MatrixSymbol, Symbol, S
from sympy.computations.matrices.blas import GEMM

def test_memtime():
    memhierarchy = [(1, 2), (.1, 4), (.01, 100000)]
    assert memtime(1, memhierarchy) == 1
    assert memtime(3, memhierarchy) == 12
    assert memtime(5, memhierarchy) == 32
    assert memtime(15, memhierarchy) == 942

def test_numelements():
    assert numelements(Symbol('x') + 3) == 1
    assert numelements(MatrixSymbol('x', 2, 3) * 2) == 6

A = MatrixSymbol('A', 3, 4)
B = MatrixSymbol('B', 4, 5)
C = MatrixSymbol('C', 3, 5)
comp = GEMM(S(13), A, B, S(14), C, 'D')
scomp = GEMM(S(13), A, B, S(14), C, 'S')

def test_nbytesoutputs():
    assert nbytesofoutputs(comp) == 3*5 * 8
    assert nbytesofoutputs(scomp) == 3*5 * 4

def test_nbytesinputs():
    assert nbytesofinputs(comp) == (3*4 + 4*5 + 3*5 + 1 + 1) * 8
    assert nbytesofinputs(scomp) == (3*4 + 4*5 + 3*5 + 1 + 1) * 4

gpuflops = 100
gpumem   = [(1, 1e9)]

cpuflops = 10
cpumem   = [(5, 2e9)]

def get_flopmem((typ, num)):
    if typ == 'cpu':
        return cpuflops, cpumem
    if typ == 'gpu':
        return gpuflops, gpumem
    raise ValueError()

def cpulatency_bandwidth(A, B):
    return 1, 2

def cpu_gpu_latency_bandwidth(A):
    return 1, 4

def joinlatencies(*lats):
    return 1./sum(1./x for x in lats)

def latencybandwidth(A, B):
    if A[0] == B[0] == 'cpu':
        return cpulatency_bandwidth(A[1], B[1])
    if A[0] == 'gpu':
        a, b = cpu_gpu_latency_bandwidth(A[1])
        c, d = latencybandwidth(('cpu',A[1]),B)
        return  (a + c, joinlatencies(b, d))
    if B[0] == 'gpu':
        a, b = cpu_gpu_latency_bandwidth(B[1])
        c, d = latencybandwidth(A,('cpu',B[1]))
        return  (a + c, joinlatencies(b, d))
    raise ValueError()

def test_latencybandwidth():
    assert latencybandwidth(('cpu', 0), ('cpu', 1)) == (1, 2)
    assert latencybandwidth(('cpu', 0), ('gpu', 1)) == (2, 2/1.5)

def test_commcost():
    n = nbytesofoutputs(comp)
    assert commcost(comp, ('cpu', 1), ('cpu', 2), latencybandwidth) == 1 + n/2.
    assert commcost(comp, ('gpu', 1), ('cpu', 2), latencybandwidth) == \
            2 + n/2. + n/4.

def test_compcost():
    A = MatrixSymbol('A', 3, 4)
    B = MatrixSymbol('B', 4, 5)
    C = MatrixSymbol('C', 3, 5)
    comp = GEMM(S(13), A, B, S(14), C, 'D')
    n = nbytesofinputs(comp)
    # memory bound
    assert compcost(comp, ('gpu', 1), get_flopmem) == 1.*n / gpumem[0][0]

    A = MatrixSymbol('A', 3000, 4000)
    B = MatrixSymbol('B', 4000, 5000)
    C = MatrixSymbol('C', 3000, 5000)
    comp = GEMM(S(13), A, B, S(14), C, 'D')
    n = nbytesofinputs(comp)
    f = flops(comp)
    # compute bound
    assert compcost(comp, ('gpu', 1), get_flopmem) == 1.*flops(comp) / gpuflops
