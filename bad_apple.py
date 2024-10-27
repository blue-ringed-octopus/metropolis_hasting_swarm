# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 23:19:22 2024

@author: hibad
"""

import numpy as np
import cv2
from numba import cuda
TPB = 32

@cuda.jit()
def mh_kernel(d_out, d_epsilon, d_cov, d_normal, d_p, d_mu):
    i = cuda.grid(1)
    n = d_out.shape[0]
    if i < n:
        nmu = d_mu[i, 0]*d_normal[i, 0]+d_mu[i, 1] * \
            d_normal[i, 1]+d_mu[i, 2]*d_normal[i, 2]
        npoint = d_p[i, 0]*d_normal[i, 0]+d_p[i, 1] * \
            d_normal[i, 1]+d_p[i, 2]*d_normal[i, 2]
        ncn = d_cov[i, 0, 0]*d_normal[i, 0]**2 + 2*d_cov[i, 0, 1]*d_normal[i, 0]*d_normal[i, 1] + 2*d_cov[i, 0, 2]*d_normal[i, 0] * \
            d_normal[i, 2] + d_cov[i, 1, 1]*d_normal[i, 1]**2 + 2*d_cov[i, 1, 2] * \
            d_normal[i, 1]*d_normal[i, 2] + d_cov[i, 2, 2]*d_normal[i, 2]**2
        
      

        d_out[i, 0] = d0
        d_out[i, 1] = d1


def get_md_par(points, mu, epsilon, cov, normal):
    n = points.shape[0]
    mu = np.ascontiguousarray(mu)
    d_mu = cuda.to_device(mu)
    d_cov = cuda.to_device(cov)
    d_normal = cuda.to_device( np.ascontiguousarray(normal))
    d_p = cuda.to_device(points)
    thread = TPB
    d_out = cuda.device_array((n, 2), dtype=(np.float64))
    blocks = (n+TPB-1)//TPB
    md_kernel[blocks, thread](d_out, epsilon, d_cov, d_normal, d_p, d_mu)
    return d_out.copy_to_host()

    

cap = cv2.VideoCapture('bad_apple.mp4')
frames= []
while(cap.isOpened()):
    ret, frame = cap.read()
    if frame is None:
        break
    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    h,w = frame.shape
    frame = frame[100:h-100,200:w-200]
    frame = cv2.resize(frame,(int(w/20),int(h/20)))
    for i in range(30):
        frames.append(frame)
   # cv2.imshow('frame',frame)
    
cap.release()
# for frame in frames:
#     cv2.imshow('frame',frame)
#     cv2.waitKey(30)
# cv2.destroyAllWindows()

#%%
def f(x, phi):
    h,w = phi.shape
    if x[0]<0 or x[0]>=w or \
       x[1]<0 or x[1]>=h:
        return 0
    
    i = int(x[0])
    j = int(x[1])
 
    return phi[-j,i]
    

    
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
    
def metropolis_hasting(x, phi):
    x_prime = g_sample(x)
    g1 = g(x_prime,x)
    g2 = g(x,x_prime)
    A = np.min([1, (f(x_prime, phi)*g1)/((f(x[i,:], phi)*g2))])
    if np.random.rand() <= A:
        x = x_prime
    return x

dx = 1 
u_max = [5, np.pi]
NUM_AGENT = 200
h,w = frames[0].shape

x = np.random.randint([0,0],[w,h], (NUM_AGENT,2))
x = x.astype(np.float32)
s = [x.copy()]
pis = []
for i in range(len(frames)):
    pi = (255-frames[i])/np.sum((255-frames[i]))
    pis.append(pi.copy())
    for i in range(NUM_AGENT):
        x_prime = g_sample(x[i,:])
        g1 = g(x_prime,x[i,:])
        g2 = g(x[i,:],x_prime)
        A = np.min([1, (f(x_prime, pi)*g1)/((f(x[i,:], pi)*g2))])
        if np.random.rand() <= A:
            x[i,:]=x_prime
    s.append(x.copy())
   
s = np.array(s)    

#%%
import matplotlib.pyplot as plt 
    
# plt.figure()    
# ax = plt.gca()
# ax.set_aspect('equal', adjustable='box')
# plt.axis([0, w, 0, h])

# for i in range(int(len(s)/10)):
#     # plt.contourf(X,Y, pis[i*10], extent=[0,w, 0,h], cmap='Greys', levels = 100)
#     # plt.imshow(pis[i*10], extent=[0,w, 0,h], cmap='Greys', alpha = 0.1)
#     # plt.plot(s[:i*10,:, 0], s[:i*10,:, 1], marker="." ,color='k', alpha=0.01, markersize = 10)
#     plt.plot(s[i*10,:,0], s[i*10,:,1], ".", color="black",  markersize = 15, alpha=0.5)
#     plt.axis('square')
#     plt.xlim(0 , w)
#     plt.ylim(0, h)
#     plt.pause(0.01)
#     plt.close()  
    
 #%%
from matplotlib.animation import FuncAnimation, PillowWriter

fig = plt.figure()    
ax = plt.gca()
ax.set_aspect('equal', adjustable='box')
plt.axis([0, w, 0, h])
plt.axis('square')
plt.ylim(0, h)
plt.xlim(0 , w)
def animate(i):
    ax.clear()
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    swarm = ax.plot(s[i,:,0], s[i,:,1], ".", color="black",  markersize = 15, alpha = 0.5)
   
    return swarm
        
ani = FuncAnimation(fig, animate, interval=40, blit=True, repeat=True, frames=int(len(s)))    
ani.save("bad_apple_swarm.mp4", dpi=500,  writer='ffmpeg')   