import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define a simple cube (8 vertices)
verts = np.array([
    [-0.5,-0.5,-0.5], [0.5,-0.5,-0.5], [0.5,0.5,-0.5], [-0.5,0.5,-0.5],
    [-0.5,-0.5,0.5], [0.5,-0.5,0.5], [0.5,0.5,0.5], [-0.5,0.5,0.5]
], dtype=float)
# Define edges for a cube for plotting
edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]

def rotation_matrix(ax, ay, az):
    sx, cx = np.sin(ax), np.cos(ax)
    sy, cy = np.sin(ay), np.cos(ay)
    sz, cz = np.sin(az), np.cos(az)
    Rx = np.array([[1,0,0],[0,cx,-sx],[0,sx,cx]])
    Ry = np.array([[cy,0,sy],[0,1,0],[-sy,0,cy]])
    Rz = np.array([[cz,-sz,0],[sz,cz,0],[0,0,1]])
    return Rz @ Ry @ Rx

def transform(verts, R, t):
    return (verts @ R.T) + t

# Animation loop (save frames for quick view)
frames = 60
translations = np.linspace([0,0,0], [2,1,0.5], frames)
angles = np.linspace(0, np.pi*2, frames)

fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(-1, 3)
ax.set_ylim(-1, 2)
ax.set_zlim(-1, 1)
ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')

for i in range(frames):
    ax.clear()
    R = rotation_matrix(0, angles[i], angles[i]/2)
    t = translations[i]
    cube = transform(verts, R, t*np.array([1,1,0.5]))
    for a,b in edges:
        p1 = cube[a]; p2 = cube[b]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color='b')
    ax.set_xlim(-1, 4); ax.set_ylim(-1, 3); ax.set_zlim(-1, 2)
    plt.pause(0.05)
plt.show()
