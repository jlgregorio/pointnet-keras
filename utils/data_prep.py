import numpy as np

from utils import load_off, Mesh


def preprocess_mesh(mesh_file, num_points=2048, compute_normals=False, augment_data=False):
    """From a mesh file to a usable pointcloud."""

    # Load mesh
    vertices, faces = load_off(mesh_file)
    mesh = Mesh(vertices, faces)
    
    # Sample mesh to obtain a pointcloud
    points, sampled_faces = mesh.sample(num_points)

    # Normalize points
    points = norm_points(points)
    
    # Compute unit normal vector associated with each point
    if compute_normals:
        normals = mesh.face_normals[sampled_faces]
        points = np.hstack([points, normals])

    # Augment data
    if augment_data:
        points = augment(points)

    return points


def augment(points):
    """Data augmentation: rotate, jitter and shuffle points."""

    # Check if normals are also provided (x, y, z, nx, ny, nz)
    _, dim = points.shape
    if dim==6:
        points, normals = points[:,:3], points[:,3:]

    # Randomly rotate points along the up-axis
    theta = np.random.uniform(0., 2*np.pi)
    R_z = np.array([[np.cos(theta), -np.sin(theta), 0.],
                    [np.sin(theta), np.cos(theta), 0.],
                    [0., 0., 1.]])
    points = (R_z @ points.T).T

    if dim==6:
        normals = (R_z @ normals.T).T

    # Jitter points
    points += np.random.normal(scale=0.02, size=points.shape)
    
    # Shuffle points
    if dim==6:
        points = np.hstack([points, normals])
    np.random.shuffle(points) # inplace (return None)
    
    return points


def norm_points(points):
    """Normalize points into a unit sphere."""
    
    points -= points.mean(axis=0)
    dists = np.linalg.norm(points, axis=1)
    
    return points / np.max(dists)
