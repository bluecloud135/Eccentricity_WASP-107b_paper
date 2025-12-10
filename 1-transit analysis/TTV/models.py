import numpy as np

def ModelFunc(kind, fixed_params=None):
    if kind == 'Linear':
        Model = linearModel
        print('Using Linear model')
    elif kind == 'Pdot':
        Model = pdotModel
        print('Using Pdot model')
    elif kind == 'Precession':
        Model = apsidalPrecessionModel
        print('Using Precession model')
    elif kind == 'Sine':
        Model = sineModel
        print('Using Sine model')
    elif kind == 'SineSimple':
        Model = sineModel
        print('Using Sine Simple model')
    else:
        raise ValueError("Not a valid kind of model")
    return Model

# def linearModel(params,N):
#     t0, period = params
#     t_tras = [np.nan]*(N+1)
#     t_tras[0] = t0
#     for i in range(1,N+1):
#         t_tras[i] = t_tras[0] + period*i
#     return t_tras

def linearModel(params,N):
    t0, period = params
    t_tras = [np.nan]*(N+1)
    t_tras[0] = t0
    for i in range(1,N+1):
        t_tras[i] = t_tras[i-1] + period
    return t_tras

def pdotModel(params,N):
    factor = 24*60*60*1000*365
    t0, period0, pdot = params
    pdot /= factor
    t_tras = [np.nan]*(N+1)
    period = [np.nan]*(N+1)
    t_tras[0] = t0
    period[0] = period0
    for i in range(1,N+1):
        period[i]=(2+pdot)/(2-pdot)*period[i-1]
        t_tras[i] = t_tras[i-1] + period[i-1]
    return t_tras

def apsidalPrecessionModel(params,N):
    t0, period, e, omega_0, d_omega_dN = params
    t_tras = [np.nan]*(N+1)
    omegas = [np.nan]*(N+1)
    t_tras[0] = t0
    omegas[0] = omega_0
    P_ano = period/(1-d_omega_dN/(2*np.pi))
    for i in range(1,(N+1)):
        omegas[i] = (omegas[i-1] + d_omega_dN) % (2*np.pi)
        t_tras[i] = t_tras[i-1] + period + e*P_ano/(np.pi)*d_omega_dN*np.sin(omegas[i])
    return t_tras

def sineModel(params,N):
    t0, period, period_sine, phi, A = params
    A /= 24*60*60
    t_tras = [np.nan]*(N+1)
    phase = [np.nan]*(N+1)
    t_tras[0] = t0
    phase[0] = -phi
    dphase = 2*np.pi/period_sine
    for i in range(1,(N+1)):
        phase[i] = phase[i-1] + dphase
        t_tras[i] = t_tras[i-1] + period + A*dphase*np.cos(phase[i])
    return t_tras



# def sineModel(params,N):
#     t0, period, A, f, phi = params
#     return A*np.sin(2*np.pi*f*np.arange(N+1) - phi)