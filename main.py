import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.linalg import solve
from scipy.linalg import null_space


# This as to add transparent patches in figures
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

class _vis:
    def __init__(self):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')        
        #self.plot_coordinate_system()
        
        self.cam_centers = {}
        self.img_corners = {}
        self.__cam_counter = 0
        
        
        # Store the intersection points
        self.intersections = {}
        
        
    def plot_coordinate_system(self):
        length = 1
        self.ax.plot([0, length], [0, 0], [0, 0], "--", color='black')  # X-axis
        self.ax.plot([0, 0], [0, length], [0, 0], "--", color='black')  # Y-axis
        self.ax.plot([0, 0], [0, 0], [0, length], "--", color='black')  # Z-axis

        
    def add_cam(self,cam_center,R_c2g):
        w = 2
        h = 2
        corners = np.array([
            1, -0.5*w, 0.5*h,
            1, 0.5*w, 0.5*h,
            1, 0.5*w, -0.5*h,
            1, -0.5*w, -0.5*h,
            1, -0.5*w, 0.5*h
        ]).reshape(5,3)
        cam_center = cam_center.reshape(3,1)
        
        # First rotate the corners
        corners = corners @ R_c2g.transpose()
        
        # Now translate the corners
        for col_idx in range(3):
            corners[:,col_idx] = corners[:,col_idx] + cam_center[col_idx,0]
            
        self.ax.plot(corners[:, 0], corners[:, 1], corners[:, 2], color='b')
        # Create a Poly3DCollection with the specified vertices and faces
        poly3d = Poly3DCollection([corners], alpha=0.2, facecolors='cyan', edgecolors='k')

        # Add the collection to the 3D plot
        self.ax.add_collection3d(poly3d)

        self.ax.scatter3D(cam_center[0,0], cam_center[1,0], cam_center[2,0], color='r')
        self.ax.text(cam_center[0,0], cam_center[1,0], cam_center[2,0]+0.2, r'$\mathcal{C}$'+f'$_{self.__cam_counter}$', color='black', fontsize = 20)
        
        # Append the cam into our vector
        self.cam_centers.update({self.__cam_counter : cam_center})
        self.img_corners.update({self.__cam_counter : corners})
        
        # Increase the counter
        self.__cam_counter += 1
        
    def add_landmark(self,t_in_g):
        t_in_g = t_in_g.reshape(3,1)
        
        # Show the landmark
        self.ax.scatter3D(t_in_g[0,0], t_in_g[1,0], t_in_g[2,0], color='green')
        
        # Plot a line between the cam centers and the landmark
        for cam_idx in range(self.__cam_counter):
            cam_center = self.cam_centers[cam_idx]
            x_points = [t_in_g[0,0], cam_center[0,0]]
            y_points = [t_in_g[1,0], cam_center[1,0]]
            z_points = [t_in_g[2,0], cam_center[2,0]]
            self.ax.plot(x_points,y_points,z_points, color='green')
            
            # Find the plane Equation
            corners =  self.img_corners[cam_idx]
            corners_homogenous = np.ones((5,4))
            corners_homogenous[:,0:3] = corners
            plane_homogenous_coordinates = null_space(corners_homogenous)
            plane_homogenous_coordinates = plane_homogenous_coordinates / plane_homogenous_coordinates[3,0]
            
            # Homogeneous coordinates of center
            center_homogeneous = np.ones((4,1))
            center_homogeneous[0:3,:] = cam_center.reshape(3,1)
            
            # This is the vector from cam center to point
            m = t_in_g - cam_center
            m_homogeneous = np.ones((4,1))
            m_homogeneous[0:3,:] = m.reshape(3,1)
            
            
            # Now we are ready to compute the intersection of the img plane and the projection line
            t = - np.dot(plane_homogenous_coordinates.transpose(), center_homogeneous) / ( np.dot(plane_homogenous_coordinates.transpose(), m_homogeneous) -  plane_homogenous_coordinates[3,0])
            intersection = cam_center + t * m
            intersection = intersection.reshape(3,1)
            self.ax.scatter3D(intersection[0,0], intersection[1,0], intersection[2,0], marker=(5, 2), color='green', s = 50)
            
            self.intersections.update({cam_idx : intersection})
    
    def add_noisy_projections(self):
        # Plot a line between the cam centers and the landmark
        for cam_idx in range(self.__cam_counter):
            cam_center = self.cam_centers[cam_idx]
            intersection = self.intersections[cam_idx]
            
            # First find two vectors lying in image plane
            corners = self.img_corners[cam_idx]
            
            vec1 = corners[0,:] - corners[1,:]
            vec2 = corners[1,:] - corners[2,:]
            
            # Normalize vectors 
            vec1 = vec1 / np.linalg.norm(vec1) / 2
            vec2 = vec2 / np.linalg.norm(vec2) / 2
            
            # Randomly scale the vectors
            vec1 = vec1.reshape(3,1) * np.random.rand() 
            vec2 = vec2.reshape(3,1) * np.random.rand() 
            
            # Now find the noisy intersection point
            noisy_intersection = intersection + vec1 + vec2
            
            # Now draw noisy intersection
            self.ax.scatter3D(noisy_intersection[0,0], noisy_intersection[1,0], noisy_intersection[2,0], marker=(5, 2), color='red', s = 50)
            
            # Draw noisy lines
            line_from_center_to_noisy_intersection = noisy_intersection - cam_center
            line_to_noisy_intersection = cam_center + line_from_center_to_noisy_intersection * 3
            x_points = [cam_center[0,0], line_to_noisy_intersection[0,0]]
            y_points = [cam_center[1,0], line_to_noisy_intersection[1,0]]
            z_points = [cam_center[2,0], line_to_noisy_intersection[2,0]]
            self.ax.plot(x_points,y_points,z_points, color='black')
            
            # Draw the reprojection error
            x_points = [intersection[0,0], noisy_intersection[0,0]]
            y_points = [intersection[1,0], noisy_intersection[1,0]]
            z_points = [intersection[2,0], noisy_intersection[2,0]]
            self.ax.plot(x_points,y_points,z_points, color='red')
            
            
    def get_plot(self):
        #self.ax.set_box_aspect([1, 1, 1])
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        # Set the limits for each axis
        self.ax.set_xlim([-2, 4])
        self.ax.set_ylim([-2, 4])
        self.ax.set_zlim([-2, 4])
        
        # Disable the grids
        self.ax.grid(False)
        
        # Hide the axes
        self.ax.set_axis_off()

        plt.show()
        


if __name__ == "__main__":
    vis = _vis()
    
    cam_center = np.array([0,0,2])
    R_c2g = R.from_euler("xyz",[0,-10,45], degrees=True).as_matrix()
    vis.add_cam(cam_center,R_c2g)
    
    
    cam_center = np.array([0,4,2])
    R_c2g = R.from_euler("xyz",[0,-10,-45], degrees=True).as_matrix()
    vis.add_cam(cam_center,R_c2g)
    
    landmark1 = np.array([2,2,3])
    vis.add_landmark(landmark1)
    
    vis.add_noisy_projections()
    
    vis.get_plot()
    
    
# Camera pose (position and orientation)
# camera_position = np.array([0, 0, 0])
# camera_orientation = np.eye(3)  # Identity matrix for simplicity

# # Camera parameters
# focal_length = 1.0

# # Ray parameters
# ray_direction = np.array([1, 1, 1])

# # Plot camera pose
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# # Plot camera
# ax.scatter(*camera_position, c='r', marker='o', label='Camera')

# # Plot camera pose (axis lines)
# axis_length = 2.0
# for i in range(3):
#     axis_vector = axis_length * camera_orientation[:, i]
#     ax.quiver(*camera_position, *axis_vector, color='g', label=f'Axis {i + 1}')

# # Plot ray
# ray_endpoint = camera_position + focal_length * ray_direction
# ax.quiver(*camera_position, *ray_endpoint, color='b', label='Ray')

# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# ax.legend()

# plt.show()