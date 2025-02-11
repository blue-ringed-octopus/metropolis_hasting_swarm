# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 20:40:25 2024

@author: hibad
"""
import numpy as np
from numba import cuda
from numba.cuda import random
TPB = 32

import math 
@cuda.jit(device=True)
def d_g(x1, x2, u_max):
    
    dx = x2[0]-x1[0] 
    dy = x2[1]-x1[1]
    # dx = x2-x1
    theta = math.atan2(dy, dx)
    v = math.sqrt(dx*dx+dy*dy)
    if (theta> u_max[1]) or ( theta< -u_max[1]) :
        return 0
    elif v>=u_max[0]:
        return 0
    else:
        return 1/(u_max[1]) * 1/u_max[0]
    
@cuda.jit(device=True)
def d_f(x, dx, phi):
    h,w = phi.shape
    if x[0]<0 or x[0]>=w or \
       x[1]<0 or x[1]>=h:
        return 0
    
    i = int(x[0]*dx[0])
    j = int(x[1]*dx[1])
 
    return phi[j,i]

@cuda.jit()
def mh_kernel(d_out, d_phi, d_x, d_dx, d_u_max, rng_states):
    i = cuda.grid(1)
    n = d_out.shape[0]
    if i < n:
        x = d_x[i,:]
        theta = random.xoroshiro128p_uniform_float64(rng_states,i)*2*(d_u_max[1]) - d_u_max[1]
        v = random.xoroshiro128p_uniform_float64(rng_states, i)*d_u_max[0]
        x_prime = (x[0]+v*math.cos(theta),x[1]+v*math.sin(theta) )
        g1 = d_g(x_prime,x, d_u_max)
        g2 = d_g(x,x_prime, d_u_max)
    
        A = min(1, (d_f(x_prime, d_dx, d_phi)*g1)/((d_f(x, d_dx, d_phi)*g2)))
        if random.xoroshiro128p_uniform_float64(rng_states,i) <= A:
            x0 = x_prime[0]
            x1 = x_prime[1]
        else:
            x0 = x[0]
            x1 = x[1]
            
        d_out[i, 0] = x0
        d_out[i, 1] = x1
        # d_out[i, :] = x

def metropolis_hasting_par(x, dx, phi, u_max):
    n = x.shape[0]
    d_phi = cuda.to_device(phi)
    d_x = cuda.to_device(x)
    d_u_max= cuda.to_device(u_max)
    d_dx = cuda.to_device(dx)
    thread = TPB
    d_out = cuda.device_array((n,2), dtype=(np.float64))
    # d_out =  cuda.to_device(x)
    blocks = (n+TPB-1)//TPB
    rng_states = random.create_xoroshiro128p_states(TPB * blocks, seed=np.random.randint(0,2**32/2))

    mh_kernel[blocks, thread](d_out, d_phi, d_x, d_dx, d_u_max, rng_states)
    return d_out.copy_to_host()