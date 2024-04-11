import numpy as np

def load_off(filename):
    """A very basic reader for OFF files"""

    vertices, faces = [], []

    with open(filename, 'r', encoding="utf-8") as file:

        # First line is optional and contains the letters "OFF"
        first_line = file.readline().strip()
        if first_line.startswith("OFF"):
            # Regular case
            if len(first_line)==3:
                second_line = file.readline().strip()
            # Case where the first and second lines are mixed
            else:
                second_line = first_line[3:]
        else:
            # Case where the first line is ommitted
            second_line = first_line
        
        # Second line contains the number of vertices, faces, and edges (optional)
        numbers = list(map(int, second_line.split()))
        n_v = numbers[0]
        n_f = numbers[1]
        
        # Following n_v lines contain the list of vertices
        # Each line contains the x, y, z coordinates of a vertex
        for _ in range(n_v):
            line = file.readline()
            values = list(map(float, line.strip().split()))
            vertices.append(values[:3])

        # Following n_f lines contain the list of faces
        # Each line contains the number of vertices of the face followed by the
        # indexes of the vertices (zero indexing) and RGB values (optional)
        for _ in range(n_f):
            line = file.readline()
            values = list(map(int, line.strip().split()))
            n = values.pop(0)
            if n==3: #triangle
                faces.append(values[:3])
            elif n==4: # split quad into 2 triangles
                faces.append(values[:3])
                faces.append(values[1:4])

    return np.array(vertices), np.array(faces)


class Mesh:
    """A very basic class for handling triangular meshes."""

    def __init__(self, vertices, faces):
        """Args:
            vertices: the vertices of the mesh (n x 3).
            faces: the faces of the mesh (m x 3).
        """

        self.vertices = vertices
        self.faces = faces
    
    @property
    def faces_number(self):
        """The number of faces of the mesh (m)"""

        return len(self.faces)
    
    @property
    def face_normals(self):
        """The unit normal vector of faces"""
        
        face_coords = self.vertices[self.faces]
        v1 = face_coords[:, 1] - face_coords[:, 0]
        v2 = face_coords[:, 2] - face_coords[:, 0]
        face_normals = np.cross(v1, v2)
        face_normals /= np.linalg.norm(face_normals, axis=1, keepdims=True)

        return np.nan_to_num(face_normals)

    @property
    def faces_area(self):
        """The area of faces"""

        face_coords = self.vertices[self.faces]
        v1 = face_coords[:, 1] - face_coords[:, 0]
        v2 = face_coords[:, 2] - face_coords[:, 0]

        # The norm of the vector resulting from the cross product corresponds
        # to the double of the triangle area.
        return .5 * np.linalg.norm(np.cross(v1, v2), axis=1)


    def sample(self, num_points):
        """Randomly generate points on mesh faces"""
        
        # Randomly select faces to be sampled
        # Try to sample uniformly by weighting by faces area
        faces_weight = self.faces_area / np.sum(self.faces_area)
        faces = np.random.choice(
            # choose over all faces
            self.faces_number,
            # pick n points
            size=num_points,
            # one face may be selected several times
            replace=True,
            # the larger the face the greater the probability
            p=faces_weight
        )
        
        # Generate one random point per randomly selected triangular face
        # Formula from paper "Shape distributions" by Osada et al. (2002).
        face_coords = self.vertices[self.faces][faces]
        p0 = face_coords[:, 0]
        p1 = face_coords[:, 1]
        p2 = face_coords[:, 2]
        r1 = np.random.random_sample(size=(num_points, 1))
        r2 = np.random.random_sample(size=(num_points, 1))      
        points = (1 - np.sqrt(r1)) * p0 \
                    + np.sqrt(r1) * (1 - r2) * p1 \
                    + np.sqrt(r1) * r2 * p2
        
        return points, faces
