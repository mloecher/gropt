import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def get_moments(G, T_readout, dt):
    TE = G.size*dt*1e3 + T_readout
    tINV = int(np.floor(TE/dt/1.0e3/2.0))
    GAMMA   = 42.58e3; 
    INV = np.ones(G.size)
    INV[tINV:] = -1
    Nm = 5
    tvec = np.arange(G.size)*dt
    tMat = np.zeros((Nm, G.size))
    scaler = np.zeros(Nm)
    for mm in range(Nm):
        tMat[mm] = tvec**mm
        scaler[mm] = (dt*1e3)**mm
                                 
    moments = np.abs(GAMMA*dt*tMat@(G*INV))
    return moments

def get_bval(G, T_readout, dt):
    TE = G.size*dt*1e3 + T_readout
    tINV = int(np.floor(TE/dt/1.0e3/2.0))
    GAMMA   = 42.58e3; 
    
    INV = np.ones(G.size)
    INV[tINV:] = -1
    
    Gt = 0
    bval = 0
    for i in range(G.size):
        if i < tINV:
            Gt += G[i] * dt
        else:
            Gt -= G[i] * dt
        bval += Gt*Gt*dt

    bval *= (GAMMA*2*np.pi)**2
    
    return bval

def plot_moments(G, T_readout, dt):

    TE = G.size*dt*1e3 + T_readout
    tINV = int(np.floor(TE/dt/1.0e3/2.0))
    GAMMA   = 42.58e3; 
    INV = np.ones(G.size)
    INV[tINV:] = -1
    Nm = 5
    tvec = np.arange(G.size)*dt
    tMat = np.zeros((Nm, G.size))
    for mm in range(Nm):
        tMat[mm] = tvec**mm

    moments = np.abs(GAMMA*dt*tMat@(G*INV))
    mm = GAMMA*dt*tMat * (G*INV)[np.newaxis,:]

    plt.figure()
    mmt = np.cumsum(mm[0])
    plt.plot(mmt/np.abs(mmt).max())
    mmt = np.cumsum(mm[1])
    plt.plot(mmt/np.abs(mmt).max())
    mmt = np.cumsum(mm[2])
    plt.plot(mmt/np.abs(mmt).max())
    plt.axhline(0, color='k')


def get_moment_plots(G, T_readout, dt):

    TE = G.size*dt*1e3 + T_readout
    tINV = int(np.floor(TE/dt/1.0e3/2.0))
    GAMMA   = 42.58e3; 
    INV = np.ones(G.size)
    INV[tINV:] = -1
    Nm = 5
    tvec = np.arange(G.size)*dt
    tMat = np.zeros((Nm, G.size))
    for mm in range(Nm):
        tMat[mm] = tvec**mm

    moments = np.abs(GAMMA*dt*tMat@(G*INV))
    mm = GAMMA*dt*tMat * (G*INV)[np.newaxis,:]

    out = []
    for i in range(Nm):
        mmt = np.cumsum(mm[i])
        out.append(mmt)

    return out

def plot_waveform(G, TE, T_readout, plot_moments = False, plot_eddy = False, suptitle = '', eddy_lines=[]):
    sns.set()
    sns.set_context("talk")
    
    dt = (TE-T_readout) * 1.0e-3 / G.size
    tt = np.arange(G.size) * dt * 1e3
    tINV = TE/2.0
    
    f, axarr = plt.subplots(1, 3, squeeze=False, figsize=(12,3.5))
    
    bval = get_bval(G, T_readout, dt)
    blabel = '    bval = %.0f' % bval
    if suptitle:
        f.suptitle(suptitle + blabel)
    else:
        f.suptitle(blabel)
        
    axarr[0, 0].axvline(tINV, linestyle='--', color='0.7')
    axarr[0, 0].plot(tt, G*1000)
    axarr[0, 0].set_title('Gradient')
    axarr[0, 0].set_xlabel('t [ms]')
#     axarr[0, 0].set_ylabel('G [mT/m]')
    
    mm = get_moment_plots(G, T_readout, (TE-T_readout) * 1.0e-3 / G.size)
    axarr[0, 1].axhline(linestyle='--', color='0.7')
    for i in range(3):
        mmt = mm[i]
        axarr[0, 1].plot(tt, mmt/np.abs(mmt).max())
    axarr[0, 1].set_title('Moments')
    axarr[0, 1].set_xlabel('t [ms]')
#     axarr[0, 1].set_ylabel('Moment [AU]')
    
    all_lam = np.linspace(1e-3,120,1000)
    all_e = []
    for lam in all_lam:
        lam = lam * 1.0e-3
        r = np.diff(np.exp(-np.arange(G.size+1)*dt/lam))[::-1]
        all_e.append(100*r@G)
    
    
    for e in eddy_lines:
        axarr[0, 2].axvline(e, linestyle=':', color=(0.8, 0.1, 0.1, 0.8))
    
    axarr[0, 2].axhline(linestyle='--', color='0.7')
    axarr[0, 2].plot(all_lam, all_e)
    axarr[0, 2].set_title('Eddy')
    axarr[0, 2].set_xlabel('lam [ms]')
#     axarr[0, 2].set_ylabel(' [AU]')
    
    plt.tight_layout(w_pad=0.0, rect=[0, 0.03, 1, 0.95])