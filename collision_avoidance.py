# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 15:35:24 2024

@author: hibad
"""

import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt 
from numba import cuda
from numba.cuda import random
import math
from matplotlib.animation import FuncAnimation, PillowWriter
TPB = 32

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
        
        collision = False
        for j, y in enumerate(d_x):
            if not i == j:
                d = (x0-y[0])**2 +  (x1-y[1])**2
                if d<= 0.05**2:
                    collision = True
                    break
        if collision:
            d_out[i, 0] = x[0]
            d_out[i, 1] = x[1]
        else:
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
def get_pdf(X,Y, mus, sigmas, weights):
    weights = weights/np.sum(weights)
    x = np.vstack((X.reshape(-1), Y.reshape(-1)))
    x= x.T   
    pdf = np.zeros(len(x))
    for mu, sigma, weight in zip(mus,sigmas, weights) :  
        pdf +=  weight*multivariate_normal.pdf(x, mu, cov=sigma)
        
        
    pdf /= np.sum(pdf)
    pdf = pdf.reshape(X.shape)
    return pdf

x_max = 1
dx = 250 
x = np.linspace(0,x_max, dx)
u_max = [0.01, np.pi]
X,Y = np.meshgrid(x,x)

mus = [np.array([0.5,0.5])]
sigmas = [np.array([[0.05, 0.01], [0.01, 0.02]])]
pi = get_pdf(X,Y,mus,sigmas, [0.5, 0.5])

plt.figure()
ax = plt.gca()
ax.set_aspect('equal', adjustable='box')
plt.contourf(X,Y, pi, extent=[0,x_max, 0,x_max], cmap='Greys', levels = 100)


#%%
mus = [np.array([0.8,0.4]), np.array([0.2,0.8])]
sigmas = [np.array([[0.01, 0.005], [0.005, 0.01]]), np.array([[0.01, 0.0], [0, 0.01]])]
pi = get_pdf(X,Y,mus,sigmas, [0.5, 0.5])

plt.figure()
ax = plt.gca()
ax.set_aspect('equal', adjustable='box')
plt.contourf(X,Y, pi, extent=[0,x_max, 0,x_max], cmap='Greys', levels = 100)


NUM_AGENT = 10

x = np.random.rand(NUM_AGENT, 2)
s = [x.copy()]
pis = []
T = 5000
for i in range(T):
  #  mus = [np.array([0.8,0.4]), np.array([0.2+0.001*i,0.8])]
    mus =  [np.array([0.5,0.5])]
    phi = get_pdf(X,Y,mus,sigmas, [0.5, 0.5])
    pis.append(phi.copy())
    x = metropolis_hasting_par(x, [dx,dx], phi, u_max)

    s.append(x.copy())
   
s = np.array(s)        
# plt.plot(s[:,0], s[:,1], alpha = 0.5, color = "red")
#%%

    
plt.figure()    
ax = plt.gca()
ax.set_aspect('equal', adjustable='box')
plt.axis([0, x_max, 0, x_max])

for i in range(int(len(s)/10)):
    plt.contourf(X,Y, pis[i*10], extent=[0,x_max, 0,x_max], cmap='Greys', levels = 100)
    # plt.plot(s[:i*10,:, 0], s[:i*10,:, 1], marker="." ,color='k', alpha=0.01, markersize = 10)
    plt.plot(s[:i*10,0,0], s[:i*10,0,1], ".", color="red", alpha=1/(i+1))
    plt.plot(s[i*10,0,0], s[i*10,0,1], ".", color="red",  markersize = 10)
    plt.plot(s[i*10,1:,0], s[i*10,1:,1], ".", color="blue",  markersize = 10)
    plt.axis('square')
    plt.ylim(0, x_max)
    plt.xlim(0 , x_max)
    plt.pause(0.01)
    plt.close()
#%%
# step = 2
# fig = plt.figure()    
# ax = plt.gca()
# ax.set_aspect('equal', adjustable='box')
# plt.axis([0, x_max, 0, x_max])
# plt.axis('square')
# plt.ylim(0, x_max)
# plt.xlim(0 , x_max)
# def animate(i):
#     ax.clear()
#     ax.set_xlim(0, x_max)
#     ax.set_ylim(0, x_max)
#     dist = ax.contourf(X,Y, pis[i*step], extent=[0,x_max, 0,x_max], cmap='Greys', levels = 100)
#     swarm = ax.plot(s[i*step,:,0], s[i*step,:,1], ".", color="red",  markersize = 10)
   
#     return swarm+[dist]
        
# ani = FuncAnimation(fig, animate, interval=40, blit=True, repeat=True, frames=len(s)//step)    
# # ani.save("metropolis_hasting_swarm.gif", dpi=300, writer=PillowWriter(fps=25))
# ani.save("chemotaxis.gif", dpi=300, writer=PillowWriter(fps=25))