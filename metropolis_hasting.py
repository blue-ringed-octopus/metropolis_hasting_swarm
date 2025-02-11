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
from metropolis_hasting_par import metropolis_hasting_par
from matplotlib.animation import FuncAnimation, PillowWriter

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

def f(x, pdf):
    if x[0]<0 or x[0]>=x_max or \
       x[1]<0 or x[1]>=x_max:
        return 0
    
    i = int(x[0]*(dx/x_max))
    j = int(x[1]*(dx/x_max))
 
    return pdf[j,i]
    

    
def g_sample(x):
    theta = np.random.rand()*2*(u_max[1]) - u_max[1]
    v = np.random.rand()*u_max[0]
    x_new = x+v*np.array([np.cos(theta), np.sin(theta)])
    return x_new
    
def g(x1,x2): 
    dx = x2-x1
    theta = np.arctan2(dx[1], dx[0])
    v = np.linalg.norm(dx)
    if abs(theta)> u_max[1]:
        return 0
    elif v>=u_max[0]:
        return 0
    else:
        return 1/(u_max[1]) * 1/u_max[0]

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

#%% single agent 
x = np.random.rand(2)
s = [x.copy()]
pis = []
for i in range(50000):
    pi = get_pdf(X,Y,mus,sigmas, [0.5, 0.5])
    pis.append(pi.copy())
    x_prime = g_sample(x)
    g1 = g(x_prime,x)
    g2 = g(x,x_prime)
    A = np.min([1, (f(x_prime, pi)*g1)/((f(x, pi)*g2))])
    if np.random.rand() <= A:
        x=x_prime
    s.append(x.copy())
   
s = np.array(s)    
#%%    
fig = plt.figure()    
ax = plt.gca()
ax.set_aspect('equal', adjustable='box')
plt.axis([0, x_max, 0, x_max])
plt.axis('square')
plt.ylim(0, x_max)
plt.xlim(0 , x_max)
def animate(i):
    ax.clear()
    ax.set_xlim(0, x_max)
    ax.set_ylim(0, x_max)
    dist = ax.contourf(X,Y, pis[i], extent=[0,x_max, 0,x_max], cmap='Greys', levels = 100)
    swarm = ax.plot(s[i*10,0], s[i*10,1], ".", color="red",  markersize = 10)
    past =  ax.plot(s[:i*10,0], s[:i*10,1], ".", color="red",  markersize = 5, alpha = 0.1)
    return past+swarm+[dist]
        
ani = FuncAnimation(fig, animate, interval=40, blit=True, repeat=True, frames=int(len(s)/20))    
ani.save("metropolis_hasting_ergodic.gif", dpi=300, writer=PillowWriter(fps=25))
#%%
mus = [np.array([0.8,0.4]), np.array([0.2,0.8])]
sigmas = [np.array([[0.01, 0.005], [0.005, 0.01]]), np.array([[0.01, 0.0], [0, 0.01]])]
pi = get_pdf(X,Y,mus,sigmas, [0.5, 0.5])

plt.figure()
ax = plt.gca()
ax.set_aspect('equal', adjustable='box')
plt.contourf(X,Y, pi, extent=[0,x_max, 0,x_max], cmap='Greys', levels = 100)


NUM_AGENT = 100

x = np.random.rand(NUM_AGENT, 2)
s = [x.copy()]
pis = []
for i in range(1000):
  #  mus = [np.array([0.8,0.4]), np.array([0.2+0.001*i,0.8])]
    mus =  [np.array([0.5,0.5])]
    phi = get_pdf(X,Y,mus,sigmas, [0.5, 0.5])
    pis.append(phi.copy())
    x = metropolis_hasting_par(x, [dx,dx], phi, u_max)

    # for i in range(NUM_AGENT):
    #     x_prime = g_sample(x[i,:])
    #     g1 = g(x_prime,x[i,:])
    #     g2 = g(x[i,:],x_prime)
    #     A = np.min([1, (f(x_prime, pi)*g1)/((f(x[i,:], pi)*g2))])
    #     if np.random.rand() <= A:
    #         x[i,:]=x_prime
    s.append(x.copy())
   
s = np.array(s)        
# plt.plot(s[:,0], s[:,1], alpha = 0.5, color = "red")
#%%
# plt.figure()    
# ax = plt.gca()
# ax.set_aspect('equal', adjustable='box')
# plt.axis([0, x_max, 0, x_max])
# plt.axis('square')
# plt.ylim(0, x_max)
# plt.xlim(0 , x_max)
# for i in range(len(s)-1):
#     plt.plot(s[i,:, 0], s[i,:, 1], marker="." ,color='k', alpha=0.01, markersize = 10)
    
plt.figure()    
ax = plt.gca()
ax.set_aspect('equal', adjustable='box')
plt.axis([0, x_max, 0, x_max])

for i in range(int(len(s)/10)):
    plt.contourf(X,Y, pis[i*10], extent=[0,x_max, 0,x_max], cmap='Greys', levels = 100)
    # plt.plot(s[:i*10,:, 0], s[:i*10,:, 1], marker="." ,color='k', alpha=0.01, markersize = 10)
    plt.plot(s[i*10,:,0], s[i*10,:,1], ".", color="red",  markersize = 10)
    plt.axis('square')
    plt.ylim(0, x_max)
    plt.xlim(0 , x_max)
    plt.pause(0.01)
    plt.close()
#%%
step = 2
fig = plt.figure()    
ax = plt.gca()
ax.set_aspect('equal', adjustable='box')
plt.axis([0, x_max, 0, x_max])
plt.axis('square')
plt.ylim(0, x_max)
plt.xlim(0 , x_max)
def animate(i):
    ax.clear()
    ax.set_xlim(0, x_max)
    ax.set_ylim(0, x_max)
    dist = ax.contourf(X,Y, pis[i*step], extent=[0,x_max, 0,x_max], cmap='Greys', levels = 100)
    swarm = ax.plot(s[i*step,:,0], s[i*step,:,1], ".", color="red",  markersize = 10)
   
    return swarm+[dist]
        
ani = FuncAnimation(fig, animate, interval=40, blit=True, repeat=True, frames=len(s)//step)    
# ani.save("metropolis_hasting_swarm.gif", dpi=300, writer=PillowWriter(fps=25))
ani.save("chemotaxis.gif", dpi=300, writer=PillowWriter(fps=25))