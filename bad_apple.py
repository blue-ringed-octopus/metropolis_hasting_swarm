# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 23:19:22 2024

@author: hibad
"""
import numpy as np
import cv2
from numba import cuda
from numba.cuda import random
TPB = 32


    

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
def d_f(x, phi):
    h,w = phi.shape
    if x[0]<0 or x[0]>=w or \
       x[1]<0 or x[1]>=h:
        return 0
    
    i = int(x[0])
    j = int(x[1])
 
    return phi[-j,i]

@cuda.jit()
def mh_kernel(d_out, d_phi, d_x, d_u_max, rng_states):
    i = cuda.grid(1)
    n = d_out.shape[0]
    if i < n:
        x = d_x[i,:]
        theta = random.xoroshiro128p_uniform_float64(rng_states,i)*2*(d_u_max[1]) - d_u_max[1]
        v = random.xoroshiro128p_uniform_float64(rng_states, i)*d_u_max[0]
        x_prime = (x[0]+v*math.cos(theta),x[1]+v*math.sin(theta) )
        g1 = d_g(x_prime,x, d_u_max)
        g2 = d_g(x,x_prime, d_u_max)
    
        A = min(1, (d_f(x_prime, d_phi)*g1)/((d_f(x, d_phi)*g2)))
        if random.xoroshiro128p_uniform_float64(rng_states,i) <= A:
            x0 = x_prime[0]
            x1 = x_prime[1]
        else:
            x0 = x[0]
            x1 = x[1]
            
        d_out[i, 0] = x0
        d_out[i, 1] = x1
        # d_out[i, :] = x

def metropolis_hasting_par(x, phi, u_max):
    n = x.shape[0]
    d_phi = cuda.to_device(phi)
    d_x = cuda.to_device(x)
    d_u_max= cuda.to_device(u_max)

    thread = TPB
    d_out = cuda.device_array((n,2), dtype=(np.float64))
    # d_out =  cuda.to_device(x)
    blocks = (n+TPB-1)//TPB
    rng_states = random.create_xoroshiro128p_states(TPB * blocks, seed=np.random.randint(0,2**32/2))

    mh_kernel[blocks, thread](d_out, d_phi, d_x, d_u_max, rng_states)
    return d_out.copy_to_host()

dx = 1 
u_max = [5, np.pi]
NUM_AGENT = 200
h,w = frames[0].shape

x = np.random.randint([0,0],[w,h], (NUM_AGENT,2))
x = x.astype(np.float64)
s = [x.copy()]
for i in range(len(frames)):
# for i in range(1000):
    phi = (255-frames[i])/np.sum((255-frames[i]))
    x = metropolis_hasting_par(x, phi, u_max)
    s.append(x.copy())
   
s = np.array(s)    

#%%
import matplotlib.pyplot as plt 
    
plt.figure()    
ax = plt.gca()
ax.set_aspect('equal', adjustable='box')
plt.axis([0, w, 0, h])

for i in range(5000):
    # plt.contourf(X,Y, pis[i*10], extent=[0,w, 0,h], cmap='Greys', levels = 100)
    plt.imshow(frames[i*30], extent=[0,w, 0,h], cmap='Greys', alpha = 0.1)
    # plt.plot(s[:i*10,:, 0], s[:i*10,:, 1], marker="." ,color='k', alpha=0.01, markersize = 10)
    plt.plot(s[i*30,:,0], s[i*30,:,1], ".", color="black",  markersize = 30, alpha=0.5)
    plt.axis('square')
    plt.xlim(0 , w)
    plt.ylim(0, h)
    plt.pause(0.01)
    plt.close()  
    
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
    swarm = ax.plot(s[30*i,:,0], s[30*i,:,1], ".", color="black",  markersize = 15, alpha = 0.5)
   
    return swarm
        
ani = FuncAnimation(fig, animate, interval=40, blit=True, repeat=True, frames=int(len(s)/30))    
ani.save("bad_apple_swarm.mp4", dpi=500,  writer='ffmpeg')   