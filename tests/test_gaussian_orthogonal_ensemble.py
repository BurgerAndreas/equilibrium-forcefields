import torch




def orthogonal_matrix(dim):
    """Orthogonal matrix.
    W = V O
    O is random orthogonal matrix (Haar distributed)
    V is diagonal matrix (vector) with entries from normal distribution"""
    # a = torch.randn(dim, dim)
    # q, _ = torch.qr(a)
    # return q
    v = torch.randn(dim)
    o = orthogonal_matrix(dim)
    return torch.diag(v) @ o

def goe_matrix(dim, v=0.01, div_dim=False):
    """Gaussian orthogonal ensemble (GOE)
    The Gaussian orthogonal ensemble (GOE) is a random matrix ensemble where 
    the entries are independent Gaussian random variables with 
    mean 0 and variance V/N off-diagonal,
    mean 0 and variance 2V/N diagonal, 
    and the matrix is symmetric. 
    The eigenvalues of a GOE matrix are real and follow the Wigner semicircle distribution."""
    _div = dim if div_dim else 1.
    
    # V1 - symmetric but wrong variance
    # a = torch.normal(mean=0, std=v, size=(dim, dim))
    # return (a + a.t()) / 2
    
    # V2 - ?
    # a = torch.normal(mean=0, std=v, size=(dim, dim))
    # q, _ = torch.qr(a)
    # return (q + q.t()) / 2
    
    # V3 - 
    # H_ij ~ N(0, v)
    lower_triangular = torch.normal(mean=0, std=v/_div, size=(dim, dim))
    lower_triangular = torch.tril(lower_triangular, diagonal=-1)
    # H_ii ~ N(0, 2v)
    diagonal = torch.normal(mean=0, std=2*v/_div, size=(dim,))
    return lower_triangular + lower_triangular.t() + torch.diag(diagonal)


if __name__ == '__main__':
    N = 1000
    v = 0.01
    v = 1.
    goe = goe_matrix(N, v)
    print("is symmetric: ", torch.allclose(goe, goe.t()))
    # variance along diagonal
    print("variance along diagonal: ", f"{goe.diag().var().item():.2e}", "?= 2*v/N =", f"{2*v/N:.2e}")
    # variance off diagonal
    print("variance off diagonal: ", f"{goe[~torch.eye(N, dtype=bool)].var().item():.2e}", "?= v/N =", f"{v/N:.2e}")