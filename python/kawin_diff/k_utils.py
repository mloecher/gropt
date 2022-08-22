import numpy as np


def get_maxwell(G, dt, TE, start_t = 0):
    
    start_ind = int(start_t/dt)
    
    if TE < 1.0:
        TE *= 1e3
    
    tINV = start_ind + int(np.floor(TE/dt/1.0e3/2.0))
    
    norm0 = np.linalg.norm(G[start_ind:tINV])
    norm1 = np.linalg.norm(G[tINV:])
    
    return norm0-norm1


def get_bval(G, dt, TE, start_t = 0):
    
    start_ind = int(start_t/dt)
    
    if TE < 1.0:
        TE *= 1e3
    
    tINV = start_ind + int(np.floor(TE/dt/1.0e3/2.0))
    GAMMA   = 42.58e3; 
    
    INV = np.ones(G.size)
    INV[tINV:] = -1
    Gt = 0
    bval = 0
    for i in range(start_ind, G.size):
        if i < tINV:
            Gt += G[i] * dt
        else:
            Gt -= G[i] * dt
        bval += Gt*Gt*dt

    bval *= (GAMMA*2*np.pi)**2
    
    return bval


def get2_eddy_mode0(G, lam, dt):
    E0 = np.zeros_like(G)
    for i in range(G.size):
        ii = float(G.size - i - 1)
        if i == 0:
            val = -np.exp(-ii*dt/lam)
        else:
            val = np.exp(-(ii+1.0)*dt/lam) - np.exp(-ii*dt/lam)
        E0[i] = -val
    
    return E0

def get2_eddy_mode1(G, lam, dt):
    E1 = np.zeros_like(G)
    
    val = 0.0
    for i in range(G.size):
        ii = float(G.size - i - 1)
        val += -np.exp(-ii*dt/lam)
    E1[0] = val * 1e3 * dt
    
    for i in range(1, G.size):
        ii = float(G.size - i)
        val = -np.exp(-ii*dt/lam)
        E1[i] = val  * 1e3 * dt
    
    return E1


def get_eddy_curves(G, dt, max_lam, n_lam):
    all_lam = np.linspace(1e-4, max_lam, n_lam)
    all_e0 = []
    all_e1 = []
    for lam in all_lam:
        lam = lam * 1.0e-3

        E0 = get2_eddy_mode0(G, lam, dt)
        all_e0.append(np.sum(E0*G))

        E1 = get2_eddy_mode1(G, lam, dt)
        all_e1.append(np.sum(E1*G))

    return all_lam, all_e0, all_e1