"""Tests for pose visualization module."""

import pytest
import numpy as np
from py_playground.visualization.pose import (
    rotation_matrix,
    transform,
    PoseVisualizer,
    CUBE_VERTS,
    CUBE_EDGES,
)


class TestRotationMatrix:
    """Tests for rotation_matrix function."""
    
    def test_returns_3x3_matrix(self):
        """Test that rotation_matrix returns a 3x3 matrix."""
        R = rotation_matrix(0, 0, 0)
        assert R.shape == (3, 3)
    
    def test_identity_rotation(self):
        """Test that zero angles produce identity matrix."""
        R = rotation_matrix(0, 0, 0)
        expected = np.eye(3)
        np.testing.assert_array_almost_equal(R, expected)
    
    def test_rotation_determinant(self):
        """Test that rotation matrix has determinant 1."""
        R = rotation_matrix(0.5, 0.3, 0.2)
        det = np.linalg.det(R)
        assert abs(det - 1.0) < 1e-10
    
    def test_rotation_orthogonality(self):
        """Test that rotation matrix is orthogonal."""
        R = rotation_matrix(0.5, 0.3, 0.2)
        RRT = R @ R.T
        expected = np.eye(3)
        np.testing.assert_array_almost_equal(RRT, expected, decimal=10)
    
    def test_x_rotation_90_degrees(self):
        """Test 90 degree rotation around X axis."""
        R = rotation_matrix(np.pi / 2, 0, 0)
        # Y should go to Z, Z should go to -Y
        v = np.array([0, 1, 0])
        rotated = R @ v
        np.testing.assert_array_almost_equal(rotated, [0, 0, 1], decimal=10)


class TestTransform:
    """Tests for transform function."""
    
    def test_identity_transform(self):
        """Test transform with identity rotation and zero translation."""
        verts = np.array([[1, 0, 0], [0, 1, 0]])
        R = np.eye(3)
        t = np.array([0, 0, 0])
        result = transform(verts, R, t)
        np.testing.assert_array_almost_equal(result, verts)
    
    def test_pure_translation(self):
        """Test transform with pure translation."""
        verts = np.array([[1, 0, 0], [0, 1, 0]])
        R = np.eye(3)
        t = np.array([1, 2, 3])
        result = transform(verts, R, t)
        expected = verts + t
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_pure_rotation(self):
        """Test transform with pure rotation."""
        verts = np.array([[1, 0, 0], [0, 1, 0]])
        R = rotation_matrix(0, 0, np.pi / 2)  # 90 degree Z rotation
        t = np.array([0, 0, 0])
        result = transform(verts, R, t)
        expected = np.array([[0, 1, 0], [-1, 0, 0]])
        np.testing.assert_array_almost_equal(result, expected, decimal=10)


class TestPoseVisualizer:
    """Tests for PoseVisualizer class."""
    
    def test_default_initialization(self):
        """Test initialization with default parameters."""
        viz = PoseVisualizer()
        assert viz.verts is not None
        assert viz.edges is not None
        assert viz.fig is None
        assert viz.ax is None
    
    def test_custom_initialization(self):
        """Test initialization with custom vertices and edges."""
        verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        edges = [(0, 1), (1, 2)]
        viz = PoseVisualizer(verts=verts, edges=edges)
        np.testing.assert_array_equal(viz.verts, verts)
        assert viz.edges == edges
    
    def test_setup_plot(self):
        """Test that setup_plot creates figure and axes."""
        viz = PoseVisualizer()
        viz.setup_plot()
        assert viz.fig is not None
        assert viz.ax is not None
    
    def test_draw_frame_creates_axes_if_none(self):
        """Test that draw_frame auto-creates axes if not set up."""
        viz = PoseVisualizer()
        viz.draw_frame()
        assert viz.ax is not None


class TestConstants:
    """Tests for module constants."""
    
    def test_cube_verts_shape(self):
        """Test that cube vertices have correct shape."""
        assert CUBE_VERTS.shape == (8, 3)
    
    def test_cube_edges_count(self):
        """Test that cube has 12 edges."""
        assert len(CUBE_EDGES) == 12
    
    def test_cube_edge_indices_valid(self):
        """Test that all edge indices are valid."""
        for a, b in CUBE_EDGES:
            assert 0 <= a < 8
            assert 0 <= b < 8


def test_module_import():
    """Test that the module can be imported."""
    import py_playground
    assert py_playground.__version__ == "0.1.0"
    assert hasattr(py_playground, 'PoseVisualizer')
