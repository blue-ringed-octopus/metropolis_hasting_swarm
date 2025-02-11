# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 23:19:22 2024

@author: hibad
"""
import numpy as np
import cv2
import matplotlib.pyplot as plt 
from metropolis_hasting_par import metropolis_hasting_par



    

cap = cv2.VideoCapture('rickroll.mp4')
frames= []
while(cap.isOpened()):
    ret, frame = cap.read()
    if frame is None:
        break
    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    h,w = frame.shape
    frame = frame[150:h-200,500:w-550]
    frame = cv2.Canny(frame, 0,50)
    frame = cv2.blur(frame, (20,20)) 
    # h,w = frame.shape
    # frame = cv2.resize(frame,(int(w/20),int(h/20)))
    for i in range(30):
        frames.append(frame)
    # if len(frames) >100:
    #     break
    
cap.release()

# plt.imshow(frames[99])
#%%


dx = 1 
u_max = [1, np.pi]
NUM_AGENT = 5000
h,w = frames[0].shape

x = np.random.randint([0,0],[w,h], (NUM_AGENT,2))
x = x.astype(np.float64)
s = [x.copy()]
for i in range(len(frames)):
# for i in range(2000):
    phi = np.ascontiguousarray(np.flip((frames[i])/np.sum((frames[i])), 0) )
    x = metropolis_hasting_par(x, [dx,dx], phi, u_max)
    s.append(x.copy())
   
s = np.array(s)    

#%%
    
# plt.figure()    
# ax = plt.gca()
# ax.set_aspect('equal', adjustable='box')
# plt.axis([0, w, 0, h])

# for i in range(5000):
#     # plt.contourf(X,Y, pis[i*10], extent=[0,w, 0,h], cmap='Greys', levels = 100)
#     plt.imshow(frames[i], extent=[0,w, 0,h], cmap='Greys', alpha = 0.1)
#     # plt.plot(s[:i*10,:, 0], s[:i*10,:, 1], marker="." ,color='k', alpha=0.01, markersize = 10)
#     plt.plot(s[i,:,0], s[i,:,1], ".", color="black",  markersize = 5, alpha=0.1)
#     plt.axis('square')
#     plt.xlim(0 , w)
#     plt.ylim(0, h)
#     plt.pause(0.01)
#     plt.close()  
    
 #%%
from matplotlib.animation import FuncAnimation, PillowWriter
step = 30
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
    swarm = ax.plot(s[step*i,:,0], s[step*i,:,1], ".", color="black",  markersize = 5, alpha = 0.1)
   
    return swarm
        
ani = FuncAnimation(fig, animate, interval=40, blit=True, repeat=True, frames=int(len(s)/step))    
ani.save("rickroll_swarm_test.mp4", dpi=500,  writer='ffmpeg')   