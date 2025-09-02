# -*- coding: utf-8 -*-
"""
Refined Cart-Pole comparison of A-PE / OID / G-PE / HYBRID with Koopman (EDMDc)
models.  This version addresses several issues identified during code review:

- correlation_dimension_D2 returns NaN and an 'ok' flag when data is
  insufficient rather than 0.0.
- collect_data_method now keeps X_hist and U_hist aligned when safety resets
  occur.
- choose_u_gpe uses a linear non-clustering penalty rather than log scale.
- edmdc_fit supports multi-input data without compressing to a single input.
- distance based estimators subsample pairwise distances to limit memory.

The script evaluates coverage metrics, dimension estimates, conditioning, and
prediction errors for the four excitation strategies.
"""

import os, json, math, time, argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ======================================================================
# Utilities
# ======================================================================

def set_seed(seed=42):
    np.random.seed(int(seed))

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

def _timestamp():
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())

def clamp(v, lo, hi):
    return np.minimum(np.maximum(v, lo), hi)

def wrap_angle(a):
    return (a + np.pi) % (2*np.pi) - np.pi

# ======================================================================
# Cart-Pole dynamics
# ======================================================================

@dataclass
class CartPoleParams:
    m_c: float = 1.0
    m_p: float = 0.1
    l: float = 0.5
    g: float = 9.81
    u_max: float = 10.0
    mu_c: float = 0.0
    mu_p: float = 0.0

def cartpole_dynamics(x, u, p: CartPoleParams):
    x1, x2, th, thd = float(x[0]), float(x[1]), float(x[2]), float(x[3])
    m_c, m_p, l, g = p.m_c, p.m_p, p.l, p.g
    total_mass = m_c + m_p
    polemass_length = m_p * l
    costh = math.cos(th); sinth = math.sin(th)
    temp = (u + polemass_length * thd * thd * sinth) / total_mass
    thdd = (g * sinth - costh * temp) / (l*(4.0/3.0 - m_p * costh * costh/total_mass))
    xdd  = temp - polemass_length*thdd*costh/total_mass
    xdd -= p.mu_c * x2
    thdd -= p.mu_p * thd
    return np.array([x2, xdd, thd, thdd], dtype=float)

def rk4_step(x, u, dt, p: CartPoleParams):
    k1 = cartpole_dynamics(x, u, p)
    k2 = cartpole_dynamics(x + 0.5*dt*k1, u, p)
    k3 = cartpole_dynamics(x + 0.5*dt*k2, u, p)
    k4 = cartpole_dynamics(x + dt*k3, u, p)
    x_next = x + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
    x_next[2] = wrap_angle(x_next[2])
    return x_next

# ======================================================================
# Polynomial dictionary
# ======================================================================

def _multi_index_up_to_degree(n_dim, degree):
    def gen(idx, remaining, current, out):
        if idx == n_dim-1:
            out.append(tuple(current + [remaining])); return
        for v in range(remaining+1):
            gen(idx+1, remaining-v, current+[v], out)
    M=[]
    for d in range(degree+1):
        gen(0, d, [], M)
    return M

def phi_terms_cartpole(degree):
    return _multi_index_up_to_degree(4, degree)

def phi_eval_batch(X, terms):
    N = X.shape[1]
    feats=[]
    for a1,a2,a3,a4 in terms:
        if a1==a2==a3==a4==0:
            feats.append(np.ones((1,N)))
        else:
            f=(X[0,:]**a1)*(X[1,:]**a2)*(X[2,:]**a3)*(X[3,:]**a4)
            feats.append(f.reshape(1,N))
    return np.vstack(feats)

def ridge_regression(Y,X,lam=1e-6):
    XT=X.T
    G=X@XT
    n=G.shape[0]
    W=(Y@XT)@np.linalg.pinv(G + lam*np.eye(n))
    return W

def edmdc_fit(X, Xn, U, degree=3, lam=1e-6):
    """Fit EDMDc model without compressing multi-input U."""
    if U.ndim==1:
        U=U.reshape(1,-1)
    terms=phi_terms_cartpole(degree)
    Z = phi_eval_batch(X, terms)
    Zn= phi_eval_batch(Xn, terms)
    m=Z.shape[0]
    Xaug = np.vstack([Z, U])
    W = ridge_regression(Zn, Xaug, lam=lam)
    A=W[:,:m]
    B=W[:,m:]
    C = ridge_regression(X, Z, lam=lam)
    Sigma_phi=(Z@Z.T)/max(Z.shape[1],1)
    eig_phi=np.linalg.eigvalsh((Sigma_phi+Sigma_phi.T)/2.0)
    lam_min_phi=float(np.maximum(np.min(eig_phi),0.0))
    cond_Z=float(np.linalg.cond(Z))
    cond_aug=float(np.linalg.cond(Xaug))
    return {"A":A,"B":B,"C":C,"terms":terms,"Z":Z,"Zn":Zn,"Xaug":Xaug,
            "lam_min_Sigma_phi":lam_min_phi,
            "cond_Z":cond_Z,"cond_aug":cond_aug}

def _apply_B(B, u):
    """Multiply input matrix B by control vector u."""
    u = np.atleast_1d(u).reshape(-1)
    if B.ndim==1:
        return B*u
    return B @ u

def simulate_edmdc_rollout(x0, U, model):
    A,B,C,terms=model["A"],model["B"],model["C"],model["terms"]
    def phi(x):
        return phi_eval_batch(x.reshape(4,1),terms).flatten()
    u_seq = np.atleast_2d(U)
    N = u_seq.shape[1]
    m=A.shape[0]
    Z=np.zeros((m,N+1)); Xh=np.zeros((4,N+1))
    Z[:,0]=phi(x0)
    Xh[:,0]=C@Z[:,0]
    for k in range(N):
        Z[:,k+1]=A@Z[:,k] + _apply_B(B, u_seq[:,k])
        Xh[:,k+1]=C@Z[:,k+1]
    return Xh

def simulate_edmdc_single_step(X_true, U, model):
    A,B,C,terms=model["A"],model["B"],model["C"],model["terms"]
    u_seq=np.atleast_2d(U)
    N=u_seq.shape[1]
    if X_true.shape[1]!=N+1:
        raise ValueError("X_true must have one more column than U.")
    Z_true=phi_eval_batch(X_true[:,:-1],terms)
    Bu=B @ u_seq
    Z_next = A @ Z_true + Bu
    X_pred = C @ Z_next
    return np.hstack([X_true[:,0:1], X_pred])

def simulate_edmdc_receding_horizon(X_true,U,model,horizon):
    u_seq=np.atleast_2d(U)
    N=u_seq.shape[1]
    preds=[]
    for k in range(0,N-horizon+1):
        x0=X_true[:,k]
        uh=u_seq[:,k:k+horizon]
        Xh=simulate_edmdc_rollout(x0,uh,model)
        preds.append(Xh[:,-1])
    return np.array(preds).T

# ======================================================================
# Dimension estimators
# ======================================================================

def _pairwise_dists_subsample(X, max_pairs=200_000, rng=None):
    if rng is None:
        rng=np.random.default_rng()
    N=X.shape[1]
    M=min(max_pairs, N*(N-1)//2)
    if M<=0:
        return np.array([])
    i=rng.integers(0,N,size=M)
    j=rng.integers(0,N,size=M)
    mask=i!=j
    i=i[mask]; j=j[mask]
    d=np.linalg.norm(X[:,i]-X[:,j], axis=0)
    return d

def lambda_min_cov_nd(X):
    if X.shape[1]<3:
        return 0.0
    Xc=X - X.mean(axis=1, keepdims=True)
    S=(Xc@Xc.T)/max(X.shape[1],1)
    w=np.linalg.eigvalsh((S+S.T)/2.0)
    return float(np.maximum(np.min(w),0.0))

def multiscale_nonclustering_ratio_nd(X, scales=(0.6,0.3,0.15), rho0=6.0, window=None):
    X=np.asarray(X)
    if window is not None and X.shape[1]>window:
        X=X[:,-window:]
    N=X.shape[1]
    if N<20:
        return {"ratios":[np.inf]*len(scales),"ok":False}
    mu=X.mean(axis=1,keepdims=True)
    std=X.std(axis=1,keepdims=True)+1e-9
    ratios=[]; ok=True
    for s in scales:
        bw=s*std
        keys=np.floor((X-mu)/bw).astype(np.int64).T
        uniq,counts=np.unique(keys,axis=0,return_counts=True)
        n_cells=max(9,len(uniq))
        expected=max(1.0,N/n_cells)
        ratio=float(counts.max()/expected)
        ratios.append(ratio)
        if ratio>rho0:
            ok=False
    return {"ratios":ratios,"ok":ok}

def _linear_fit_with_R2(x,y):
    x=np.asarray(x); y=np.asarray(y)
    A=np.vstack([x,np.ones_like(x)]).T
    coeff, *_ = np.linalg.lstsq(A,y,rcond=None)
    yhat=A@coeff
    ss_res=np.sum((y-yhat)**2)
    ss_tot=np.sum((y-y.mean())**2)+1e-12
    R2=1.0-ss_res/ss_tot
    return float(coeff[0]), float(coeff[1]), float(R2)

def box_counting_dimension_nd(X, eps_list=(1/8,1/12,1/16,1/24,1/32,1/48,1/64)):
    X=np.asarray(X); d,N=X.shape
    mins=X.min(axis=1,keepdims=True); span=X.max(axis=1,keepdims=True)-mins
    span=np.maximum(span,1e-9)
    Y=(X-mins)/span
    Ns=[]; inv_eps=[]
    for eps in eps_list:
        g=int(max(2,math.ceil(1.0/eps)))
        idx=np.floor(Y.T*g).astype(int)
        idx=np.clip(idx,0,g-1)
        uniq=np.unique(idx,axis=0)
        Ns.append(len(uniq)); inv_eps.append(1.0/eps)
    xs=np.log(inv_eps); ys=np.log(np.array(Ns)+1e-12)
    slope,intercept,R2=_linear_fit_with_R2(xs,ys)
    return float(slope),{"inv_eps":inv_eps,"N_boxes":Ns,"R2":R2}

def correlation_dimension_D2(X, sample_max=3000, n_r=15, q_range=(0.1,0.6)):
    d,N=X.shape
    idx=np.random.choice(N,size=min(N,sample_max),replace=False)
    Y=X[:,idx]
    dists=_pairwise_dists_subsample(Y,max_pairs=200000)
    dists=dists[dists>0]
    if dists.size<100:
        return np.nan,{"r_min":None,"r_max":None,"R2":0.0,"ok":False}
    r_min=np.quantile(dists,q_range[0]); r_max=np.quantile(dists,q_range[1])
    rs=np.exp(np.linspace(np.log(r_min+1e-12),np.log(r_max+1e-12),n_r))
    C=[]
    for r in rs:
        C.append(np.mean(dists<=r))
    xs=np.log(rs+1e-18); ys=np.log(np.array(C)+1e-18)
    slope, intercept, R2=_linear_fit_with_R2(xs,ys)
    return float(max(slope,0.0)),{"r_min":float(r_min),"r_max":float(r_max),"R2":float(R2),"ok":True}

def knn_intrinsic_dimension(X, k=10, sample_max=4000):
    d,N=X.shape
    idx=np.random.choice(N,size=min(N,sample_max),replace=False)
    Y=X[:,idx].T
    M=Y.shape[0]
    norms=np.sum(Y*Y,axis=1,keepdims=True)
    D2=norms+norms.T-2*(Y@Y.T)
    D2[D2<0]=0.0
    D=np.sqrt(D2+1e-18)
    sortD=np.sort(D,axis=1)[:,1:k+1]
    rk=sortD[:,-1]+1e-18
    logs=np.log(rk.reshape(-1,1)/(sortD[:,:-1]+1e-18))
    m_inv=np.mean(np.mean(logs,axis=1))
    if m_inv<=0:
        return np.nan,{"k":k,"M_used":M,"ok":False}
    m_hat=1.0/m_inv
    return float(m_hat),{"k":int(k),"M_used":int(M),"ok":True}

# ======================================================================
# Data collection utilities
# ======================================================================

# (The rest of file will continue...) 
# ============================ Data collection ============================

def multi_sine_sequence(L, num_sines=3, f_min=0.2, f_max=2.5):
    freqs=np.random.uniform(f_min,f_max,size=num_sines)
    phases=np.random.uniform(0,2*np.pi,size=num_sines)
    amps=np.random.uniform(0.2,1.0,size=num_sines)/num_sines
    t=np.arange(L); u=np.zeros(L)
    for a,f,ph in zip(amps,freqs,phases):
        u+=a*np.sin(2*np.pi*f*(t/L)+ph)
    u=clamp(u,-1.0,1.0)
    return u

def gen_reset_points_4d(n_points, p_range=(-1.5,1.5), v_range=(-2.0,2.0),
                        th_range=(-np.pi/2,np.pi/2), thd_range=(-6.0,6.0), mode="halton"):
    def halton(n,dim=4,bases=(2,3,5,7)):
        def vdc(i,b):
            f=1.0; r=0.0
            while i>0:
                f/=b; r+=f*(i% b); i//=b
            return r
        seq=np.zeros((dim,n))
        for d in range(dim):
            b=bases[d%len(bases)]
            for k in range(1,n+1):
                seq[d,k-1]=vdc(k,b)
        return seq
    if mode=="halton":
        h=halton(n_points,4)
        Pmin=np.array([p_range[0],v_range[0],th_range[0],thd_range[0]],dtype=float).reshape(4,1)
        Pmax=np.array([p_range[1],v_range[1],th_range[1],thd_range[1]],dtype=float).reshape(4,1)
        pts=Pmin+(Pmax-Pmin)*h
        return [pts[:,k] for k in range(n_points)]
    rng=np.random.default_rng(); pts=[]
    for _ in range(n_points):
        pts.append(np.array([
            rng.uniform(*p_range),
            rng.uniform(*v_range),
            rng.uniform(*th_range),
            rng.uniform(*thd_range)
        ],dtype=float))
    return pts

def weakest_direction_from_cov(X_hist):
    if len(X_hist)<10:
        v=np.random.randn(4); return v/(np.linalg.norm(v)+1e-12)
    X=np.array(X_hist).T
    Xc=X-X.mean(axis=1,keepdims=True)
    S=(Xc@Xc.T)/max(X.shape[1],1)
    w,V=np.linalg.eigh((S+S.T)/2.0)
    vmin=V[:,np.argmin(w)]
    return vmin/(np.linalg.norm(vmin)+1e-12)

class NonClusteringMonitorND:
    def __init__(self, scales=(0.6,0.3,0.15), window=5000):
        self.scales=scales; self.window=window; self.hist=[]
    def append(self,x):
        self.hist.append(np.array(x,dtype=float))
        if len(self.hist)>self.window:
            self.hist=self.hist[-self.window:]
    def ratios(self):
        if not self.hist:
            return {"ratios":[0.0]*len(self.scales),"ok":True}
        X=np.array(self.hist).T
        return multiscale_nonclustering_ratio_nd(X,scales=self.scales,rho0=6.0,window=self.window)
    def predict_ratios_with(self,pending_points):
        X=np.array(self.hist+[np.array(p) for p in pending_points]).T
        return multiscale_nonclustering_ratio_nd(X,scales=self.scales,rho0=6.0,window=self.window)

def choose_u_gpe(xk, omega, dt, p: CartPoleParams, monitor: NonClusteringMonitorND,
                 u_grid=None, L_pred=5, alpha_orth=0.15, beta_occ=0.3):
    if u_grid is None:
        u_grid=np.linspace(-p.u_max,p.u_max,25)
    cands=[]
    for u in u_grid:
        x=xk.copy(); seg=[x.copy()]
        for _ in range(L_pred):
            x=rk4_step(x,float(u),dt,p); seg.append(x.copy())
        d=x-xk
        proj=float(np.dot(omega,d))
        orth=float(np.linalg.norm(d-proj*omega))
        r_pred=monitor.predict_ratios_with(seg[1:])
        penalty=max(r_pred["ratios"])
        score=proj - alpha_orth*orth - beta_occ*penalty
        cands.append((score,float(u)))
    cands.sort(key=lambda t: t[0],reverse=True)
    return cands[0][1]

def choose_u_oid(xk, X_hist, U_hist, dt, p: CartPoleParams, u_grid=None):
    if u_grid is None:
        u_grid=np.linspace(-p.u_max,p.u_max,25)
    if len(X_hist)<5:
        return float(np.random.choice(u_grid))
    X=np.array(X_hist).T
    if len(U_hist)>0:
        U = np.array(U_hist).reshape(1,-1)
        if U.shape[1] < X.shape[1]:
            U = np.hstack([U, np.zeros((1, X.shape[1]-U.shape[1]))])
        elif U.shape[1] > X.shape[1]:
            U = U[:, :X.shape[1]]
    else:
        U = np.zeros((1, X.shape[1]))
    lam_best=-1.0; u_best=float(np.random.choice(u_grid))
    for u in u_grid:
        x_next=rk4_step(xk,float(u),dt,p)
        Xc=np.hstack([X,x_next.reshape(4,1)])
        Uc=np.hstack([U,np.array([[u]])])
        AUG=np.vstack([Xc,Uc])
        lam=lambda_min_cov_nd(AUG)
        if lam>lam_best:
            lam_best=lam; u_best=float(u)
    return u_best

def collect_data_method(method, p: CartPoleParams, dt, budget_pairs,
                        reset_every=200, reset_points=None,
                        jitter_sigma=0.2, L_pred=5,
                        monitor_scales=(0.6,0.3,0.15), seed=42):
    set_seed(seed)
    X_cols=[]; Xn_cols=[]; U_cols=[]
    monitor=NonClusteringMonitorND(scales=monitor_scales,window=5000)
    if reset_points is None or len(reset_points)==0:
        reset_points=gen_reset_points_4d(64)
    rpi=0
    x=np.array(reset_points[rpi%len(reset_points)],dtype=float)
    if method=="APE":
        u_seq=multi_sine_sequence(budget_pairs,num_sines=4)
        for k in range(budget_pairs):
            u=p.u_max*(float(u_seq[k])+np.random.normal(0.0,jitter_sigma))
            u=float(clamp(u,-p.u_max,p.u_max))
            x_next=rk4_step(x,u,dt,p)
            X_cols.append(x.copy()); Xn_cols.append(x_next.copy()); U_cols.append([u])
            monitor.append(x_next); x=x_next
    else:
        X_hist=[x.copy()]; U_hist=[]
        for t in range(budget_pairs):
            if (t%reset_every==0) and (t>0):
                rpi+=1
                x=np.array(reset_points[rpi%len(reset_points)],dtype=float)
                X_hist.append(x.copy())
                if len(U_hist) < len(X_hist)-1:
                    U_hist.append(0.0)
            if method=="OID":
                u=choose_u_oid(x,X_hist,U_hist,dt,p)
                u+=np.random.normal(0.0,jitter_sigma*p.u_max)
            elif method=="GPE":
                omega=weakest_direction_from_cov(X_hist)
                u=choose_u_gpe(x,omega,dt,p,monitor,L_pred=L_pred)
                u+=np.random.normal(0.0,jitter_sigma*p.u_max)
            elif method=="HYBRID":
                omega=weakest_direction_from_cov(X_hist)
                u_g=choose_u_gpe(x,omega,dt,p,monitor,L_pred=L_pred)
                u_o=choose_u_oid(x,X_hist,U_hist,dt,p)
                u=0.6*u_g+0.4*u_o+np.random.normal(0.0,jitter_sigma*p.u_max)
            else:
                u=float(np.random.uniform(-p.u_max,p.u_max))
            u=float(clamp(u,-p.u_max,p.u_max))
            x_next=rk4_step(x,u,dt,p)
            if abs(x_next[2])>np.pi/1.2 or abs(x_next[0])>2.5:
                rpi+=1
                x=np.array(reset_points[rpi%len(reset_points)],dtype=float)
                X_hist.append(x.copy())
                U_hist.append(u)
                continue
            X_cols.append(x.copy()); Xn_cols.append(x_next.copy()); U_cols.append([u])
            monitor.append(x_next); x=x_next
            X_hist.append(x.copy()); U_hist.append(u)
    X=np.array(X_cols).T; Xn=np.array(Xn_cols).T; U=np.array(U_cols).T
    return X,Xn,U

# =============================== Plotting ===============================

def plot_training_phase(X, fig_dir:Path, title, fname):
    plt.figure(figsize=(6,5))
    plt.scatter(X[0,:],X[2,:],s=1,alpha=0.35)
    plt.xlabel("p"); plt.ylabel("theta"); plt.title(title)
    plt.grid(True,alpha=0.3); plt.tight_layout()
    path=fig_dir/fname; plt.savefig(path,dpi=150); plt.close()
    return str(path)

def plot_rollout(X_true,X_pred,dt,fig_dir:Path,tag:str,rid:int):
    t=np.arange(X_true.shape[1])*dt
    plt.figure(figsize=(8,3))
    plt.plot(t,X_true[0,:],label="true p",lw=2)
    plt.plot(t,X_pred[0,:],"--",label="pred p",lw=2)
    plt.xlabel("t (s)"); plt.ylabel("p"); plt.legend(); plt.grid(True,alpha=0.3)
    p1=fig_dir/f"rollout_{tag}_run{rid}_p.png"; plt.savefig(p1,dpi=150); plt.close()
    plt.figure(figsize=(8,3))
    plt.plot(t,X_true[2,:],label="true theta",lw=2)
    plt.plot(t,X_pred[2,:],"--",label="pred theta",lw=2)
    plt.xlabel("t (s)"); plt.ylabel("theta"); plt.legend(); plt.grid(True,alpha=0.3)
    p2=fig_dir/f"rollout_{tag}_run{rid}_theta.png"; plt.savefig(p2,dpi=150); plt.close()
    plt.figure(figsize=(5,5))
    plt.plot(X_true[0,:],X_true[2,:],label="true",lw=2)
    plt.plot(X_pred[0,:],X_pred[2,:],"--",label="pred",lw=2)
    plt.xlabel("p"); plt.ylabel("theta"); plt.legend(); plt.grid(True,alpha=0.3)
    p3=fig_dir/f"rollout_{tag}_run{rid}_phase.png"; plt.savefig(p3,dpi=150); plt.close()
    return [str(p1),str(p2),str(p3)]

def plot_svd_spectrum(Xaug, fig_dir:Path, tag:str):
    s=np.linalg.svd(Xaug,compute_uv=False)
    cond=float(np.max(s)/np.maximum(np.min(s),1e-12))
    plt.figure(figsize=(6,3))
    plt.semilogy(s,marker='o')
    plt.title(f"SVD singular values (cond={cond:.2e}) - {tag}")
    plt.xlabel("index"); plt.ylabel("singular value"); plt.grid(True,alpha=0.3)
    p=fig_dir/f"svd_{tag}.png"; plt.savefig(p,dpi=150); plt.close()
    return str(p),cond

# =============================== Experiment ===============================

def _out_dirs(tag:str):
    base=Path("runs")/f"{_timestamp()}_{tag}"
    figd=base/"figs"; ensure_dir(figd)
    return base,figd

def run_method(args,label,base:Path,figd:Path,method,budget_pairs):
    p=CartPoleParams(); dt=args.dt
    resets=gen_reset_points_4d(n_points=max(64,budget_pairs//max(1,args.reset_every)),
                               p_range=(-1.5,1.5),v_range=(-2.5,2.5),
                               th_range=(-np.pi/2,np.pi/2),thd_range=(-8.0,8.0),
                               mode=args.reset_mode)
    if method=="APE":
        X,Xn,U=collect_data_method("APE",p,dt,budget_pairs,reset_every=args.reset_every,
                                   reset_points=resets,jitter_sigma=args.jitter_sigma,
                                   L_pred=args.L_pred,seed=args.seed)
    elif method=="OID":
        X,Xn,U=collect_data_method("OID",p,dt,budget_pairs,reset_every=args.reset_every,
                                   reset_points=resets,jitter_sigma=args.jitter_sigma,
                                   L_pred=args.L_pred,seed=args.seed)
    elif method=="GPE":
        X,Xn,U=collect_data_method("GPE",p,dt,budget_pairs,reset_every=args.reset_every,
                                   reset_points=resets,jitter_sigma=args.jitter_sigma,
                                   L_pred=args.L_pred,seed=args.seed)
    else:
        X,Xn,U=collect_data_method("HYBRID",p,dt,budget_pairs,reset_every=args.reset_every,
                                   reset_points=resets,jitter_sigma=args.jitter_sigma,
                                   L_pred=args.L_pred,seed=args.seed)
    model=edmdc_fit(X,Xn,U,degree=args.poly_degree,lam=1e-6)
    eigA=np.linalg.eigvals(model["A"]); spec_rad=float(np.max(np.abs(eigA)))
    lam_min_Sx=lambda_min_cov_nd(X)
    lam_min_Sphi=model["lam_min_Sigma_phi"]
    ncl=multiscale_nonclustering_ratio_nd(X,scales=(0.6,0.3,0.15),rho0=6.0,window=5000)
    D_box,box_info=box_counting_dimension_nd(X)
    D2,D2_info=correlation_dimension_D2(X,sample_max=3000,n_r=15,q_range=(0.1,0.6))
    Dk,Dk_info=knn_intrinsic_dimension(X,k=10,sample_max=4000)
    phase_plot=plot_training_phase(X,figd,f"Training phase ({label})",f"phase_{label}.png")
    svd_plot,cond_aug=plot_svd_spectrum(model["Xaug"],figd,tag=label)
    T_test=6.0; N_test=int(T_test/dt)
    t_idx=np.arange(N_test)
    U_test=p.u_max*0.6*np.sin(t_idx*dt*2.5)+p.u_max*0.25*np.sin(t_idx*dt*7.0)
    x0_cases=[np.array([0.0,0.0,0.1,0.0]),np.array([0.5,-0.5,-0.2,3.0]),
              np.array([-0.8,0.5,0.2,-4.0]),np.array([1.0,0.0,0.4,0.0]),
              np.array([-1.2,0.2,-0.3,1.5])]
    fig_paths=[phase_plot,svd_plot]
    rmse_rollout=[]; rmse_single=[]
    horizons=[5,10,15,20]; rmse_receding={h:[] for h in horizons}
    for i,x0 in enumerate(x0_cases, start=1):
        X_true=simulate_true(x0,U_test,dt,p)
        Xr=simulate_edmdc_rollout(x0,U_test,model)
        fig_paths+=plot_rollout(X_true,Xr,dt,figd,label,i)
        e_r=X_true-Xr
        rmse_rollout.append(float(np.sqrt(np.mean(e_r**2))))
        Xs=simulate_edmdc_single_step(X_true,U_test,model)
        e_s=X_true[:,1:]-Xs[:,1:]
        rmse_single.append(float(np.sqrt(np.mean(e_s**2))))
        for h in horizons:
            Xh=simulate_edmdc_receding_horizon(X_true,U_test,model,h)
            e_h=X_true[:,h:h+Xh.shape[1]]-Xh
            rmse_receding[h].append(float(np.sqrt(np.mean(e_h**2))))
    metrics={"label":label,
              "config":{"method":method,"dt":float(dt),"poly_degree":int(args.poly_degree),
                        "seed":int(args.seed),"budget_pairs":int(X.shape[1]),
                        "reset_mode":args.reset_mode,"reset_every":int(args.reset_every),
                        "jitter_sigma":float(args.jitter_sigma),"L_pred":int(args.L_pred)},
              "coverage":{"lambda_min_Sigma_x":float(lam_min_Sx),
                           "lambda_min_Sigma_phi":float(lam_min_Sphi),
                           "rho_multiscale":ncl["ratios"],"rho_ok":bool(ncl["ok"]),
                           "D_box":float(D_box),
                           "D_box_fit":{"inv_eps":box_info["inv_eps"],"N_boxes":box_info["N_boxes"],"R2":float(box_info["R2"])}},
              "conditioning":{"svd_cond_aug":float(cond_aug),
                               "cond_Z":float(model["cond_Z"]),
                               "spec_radius_A":float(spec_rad)},
              "figures":fig_paths,
              "shapes":{"X":list(X.shape),"U":list(U.shape)}}
    metrics["coverage"].update({"D2_corr":float(D2),
                                "D2_fit":{"r_min":D2_info["r_min"],"r_max":D2_info["r_max"],"R2":float(D2_info["R2"]),"ok":bool(D2_info.get("ok",True))},
                                "D_kNN":float(Dk),
                                "D_kNN_fit":{"k":Dk_info["k"],"M_used":Dk_info["M_used"],"ok":bool(Dk_info.get("ok",True))}})
    metrics["rmse_results"]={"rollout":{"mean":float(np.mean(rmse_rollout)),"median":float(np.median(rmse_rollout)),"all":rmse_rollout},
                              "single_step":{"mean":float(np.mean(rmse_single)),"median":float(np.median(rmse_single)),"all":rmse_single},
                              "receding_horizon":{str(h):{"mean":float(np.mean(rmse_receding[h])),
                                                           "median":float(np.median(rmse_receding[h])),
                                                           "all":rmse_receding[h]} for h in horizons}}
    with open(base/"metrics.json","w",encoding="utf-8") as f:
        json.dump(metrics,f,indent=2,ensure_ascii=False)
    return metrics,model

def simulate_true(x0,U,dt,p:CartPoleParams):
    N=len(U)
    X=np.zeros((4,N+1)); X[:,0]=x0
    for k in range(N):
        X[:,k+1]=rk4_step(X[:,k],float(U[k]),dt,p)
    return X

# =============================== CLI ===============================

def main():
    parser=argparse.ArgumentParser(description="Cart-Pole: G-PE vs A-PE/OID/HYBRID with Koopman (EDMDc)")
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--dt',type=float,default=0.02)
    parser.add_argument('--poly-degree',type=int,default=3)
    parser.add_argument('--budget-pairs',type=int,default=None)
    parser.add_argument('--reset-mode',choices=['halton','grid','random'],default='halton')
    parser.add_argument('--reset-every',type=int,default=200)
    parser.add_argument('--jitter-sigma',type=float,default=0.2)
    parser.add_argument('--L-pred',type=int,default=5)
    args=parser.parse_args()
    set_seed(args.seed)
    compare_dir=Path("runs")/f"{_timestamp()}_COMPARE"; ensure_dir(compare_dir)
    budget_pairs=30000 if args.budget_pairs is None else int(args.budget_pairs)
    base_A,fig_A=_out_dirs("APE")
    print(f"\n=== A-PE | budget={budget_pairs} ===")
    metrics_A,_=run_method(args,"APE",base_A,fig_A,method="APE",budget_pairs=budget_pairs)
    base_O,fig_O=_out_dirs("OID")
    print(f"\n=== OID | budget={budget_pairs} ===")
    metrics_O,_=run_method(args,"OID",base_O,fig_O,method="OID",budget_pairs=budget_pairs)
    base_G,fig_G=_out_dirs("GPE")
    print(f"\n=== G-PE | budget={budget_pairs} ===")
    metrics_G,_=run_method(args,"GPE",base_G,fig_G,method="GPE",budget_pairs=budget_pairs)
    base_H,fig_H=_out_dirs("HYBRID")
    print(f"\n=== HYBRID | budget={budget_pairs} ===")
    metrics_H,_=run_method(args,"HYBRID",base_H,fig_H,method="HYBRID",budget_pairs=budget_pairs)
    compare={"budget_pairs":budget_pairs,
             "APE":{"dir":str(base_A.resolve()),"coverage":metrics_A["coverage"],"conditioning":metrics_A["conditioning"],"rmse_results":metrics_A["rmse_results"]},
             "OID":{"dir":str(base_O.resolve()),"coverage":metrics_O["coverage"],"conditioning":metrics_O["conditioning"],"rmse_results":metrics_O["rmse_results"]},
             "GPE":{"dir":str(base_G.resolve()),"coverage":metrics_G["coverage"],"conditioning":metrics_G["conditioning"],"rmse_results":metrics_G["rmse_results"]},
             "HYBRID":{"dir":str(base_H.resolve()),"coverage":metrics_H["coverage"],"conditioning":metrics_H["conditioning"],"rmse_results":metrics_H["rmse_results"]}}
    with open(compare_dir/"compare_metrics.json","w",encoding="utf-8") as f:
        json.dump(compare,f,indent=2,ensure_ascii=False)
    print("\nSaved:")
    print(f" - A-PE   : {base_A.resolve()}")
    print(f" - OID    : {base_O.resolve()}")
    print(f" - G-PE   : {base_G.resolve()}")
    print(f" - HYBRID : {base_H.resolve()}")
    print(f" - Compare: {compare_dir.resolve()}")

if __name__=='__main__':
    main()
