"""3D pose visualization with rotation and translation."""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from typing import Optional, Tuple, List
import os


# Cube vertices
CUBE_VERTS = np.array([
    [-0.5, -0.5, -0.5],
    [0.5, -0.5, -0.5],
    [0.5, 0.5, -0.5],
    [-0.5, 0.5, -0.5],
    [-0.5, -0.5, 0.5],
    [0.5, -0.5, 0.5],
    [0.5, 0.5, 0.5],
    [-0.5, 0.5, 0.5]
], dtype=float)

# Cube edges
CUBE_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),  # Bottom face
    (4, 5), (5, 6), (6, 7), (7, 4),  # Top face
    (0, 4), (1, 5), (2, 6), (3, 7)   # Connecting edges
]


def rotation_matrix(ax: float, ay: float, az: float) -> np.ndarray:
    """Create a 3D rotation matrix from Euler angles (in radians).
    
    Args:
        ax: Rotation around X-axis
        ay: Rotation around Y-axis
        az: Rotation around Z-axis
    
    Returns:
        3x3 rotation matrix
    """
    sx, cx = np.sin(ax), np.cos(ax)
    sy, cy = np.sin(ay), np.cos(ay)
    sz, cz = np.sin(az), np.cos(az)
    
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    
    return Rz @ Ry @ Rx


def transform(verts: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Apply rotation and translation to vertices.
    
    Args:
        verts: Vertices to transform (N, 3)
        R: Rotation matrix (3, 3)
        t: Translation vector (3,)
    
    Returns:
        Transformed vertices
    """
    return (verts @ R.T) + t


class PoseVisualizer:
    """Interactive 3D pose visualizer with animation support."""
    
    def __init__(self, verts: Optional[np.ndarray] = None, edges: Optional[List[Tuple[int, int]]] = None):
        """Initialize the visualizer.
        
        Args:
            verts: Custom vertices (default: cube)
            edges: Custom edges (default: cube edges)
        """
        self.verts = verts if verts is not None else CUBE_VERTS.copy()
        self.edges = edges if edges is not None else CUBE_EDGES.copy()
        self.fig: Optional[plt.Figure] = None
        self.ax: Optional[Axes3D] = None
        
    def setup_plot(self, figsize: Tuple[int, int] = (10, 8), 
                   xlim: Tuple[float, float] = (-2, 4),
                   ylim: Tuple[float, float] = (-2, 4),
                   zlim: Tuple[float, float] = (-2, 2)) -> None:
        """Set up the 3D plot."""
        self.fig = plt.figure(figsize=figsize)
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_xlim(*xlim)
        self.ax.set_ylim(*ylim)
        self.ax.set_zlim(*zlim)
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.set_title('3D Pose Visualization')
        
    def draw_frame(self, rotation: Tuple[float, float, float] = (0, 0, 0),
                   translation: np.ndarray = np.array([0, 0, 0]),
                   clear: bool = True) -> None:
        """Draw a single frame.
        
        Args:
            rotation: Euler angles (rx, ry, rz) in radians
            translation: Translation vector [x, y, z]
            clear: Whether to clear previous frame
        """
        if self.ax is None:
            self.setup_plot()
            
        if clear:
            self.ax.clear()
            self.ax.set_xlim(-2, 4)
            self.ax.set_ylim(-2, 4)
            self.ax.set_zlim(-2, 2)
            self.ax.set_xlabel('X')
            self.ax.set_ylabel('Y')
            self.ax.set_zlabel('Z')
            
        R = rotation_matrix(*rotation)
        transformed = transform(self.verts, R, translation)
        
        for a, b in self.edges:
            p1 = transformed[a]
            p2 = transformed[b]
            self.ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], 
                        'b-', linewidth=2)
        
        # Draw vertices
        self.ax.scatter(transformed[:, 0], transformed[:, 1], transformed[:, 2], 
                       c='red', s=50)
        
    def animate(self, frames: int = 60, interval: int = 50,
                save_path: Optional[str] = None) -> FuncAnimation:
        """Create an animation of rotating cube.
        
        Args:
            frames: Number of frames
            interval: Milliseconds between frames
            save_path: Optional path to save video
            
        Returns:
            Animation object
        """
        self.setup_plot()
        
        translations = np.linspace([0, 0, 0], [2, 1, 0.5], frames)
        angles = np.linspace(0, np.pi * 2, frames)
        
        def update(frame):
            self.draw_frame(
                rotation=(0, angles[frame], angles[frame] / 2),
                translation=translations[frame],
                clear=True
            )
            return []
        
        anim = FuncAnimation(self.fig, update, frames=frames, 
                           interval=interval, blit=True)
        
        if save_path:
            anim.save(save_path, writer='ffmpeg', fps=30)
            print(f"Animation saved to {save_path}")
            
        return anim
        
    def show(self) -> None:
        """Display the plot."""
        if self.fig is None:
            self.setup_plot()
        plt.show()
        
    def save_video(self, output_path: str, frames: int = 60, fps: int = 30) -> str:
        """Save animation as video file.
        
        Args:
            output_path: Path to save the video
            frames: Number of frames
            fps: Frames per second
            
        Returns:
            Path to saved video
        """
        self.setup_plot()
        
        from matplotlib.animation import FFMpegWriter
        
        translations = np.linspace([0, 0, 0], [2, 1, 0.5], frames)
        angles = np.linspace(0, np.pi * 2, frames)
        
        writer = FFMpegWriter(fps=fps, metadata=dict(artist='PyPlayground'))
        
        with writer.saving(self.fig, output_path, dpi=100):
            for i in range(frames):
                self.draw_frame(
                    rotation=(0, angles[i], angles[i] / 2),
                    translation=translations[i],
                    clear=True
                )
                writer.grab_frame()
                
        print(f"Video saved to: {output_path}")
        return output_path


def demo() -> None:
    """Run a simple demo visualization."""
    viz = PoseVisualizer()
    viz.setup_plot()
    viz.draw_frame(rotation=(0.5, 0.3, 0.2), translation=np.array([1, 0.5, 0]))
    viz.show()


if __name__ == "__main__":
    demo()
