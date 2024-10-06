import numpy as np
from vispy import app, scene
from vispy.scene import visuals

class CustomAxis:
    def __init__(self, parent):
        self._xaxis = visuals.Line(color='red', width=4, method='gl')
        self._yaxis = visuals.Line(color='green', width=4, method='gl')
        self._zaxis = visuals.Line(color='blue', width=4, method='gl')
        self._xticks, self._yticks, self._zticks = self._create_ticks()
        self._update()

        parent.add(self._xaxis)
        parent.add(self._yaxis)
        parent.add(self._zaxis)
        parent.add(self._xticks)
        parent.add(self._yticks)
        parent.add(self._zticks)

    def _create_ticks(self):
        xticks = visuals.Line(color='black', width=2, method='gl')
        yticks = visuals.Line(color='black', width=2, method='gl')
        zticks = visuals.Line(color='black', width=2, method='gl')

        return xticks, yticks, zticks

    def _update(self):
        x_data = np.array([[0, 0, 0], [10000, 0, 0]], dtype=np.float32)
        y_data = np.array([[0, 0, 0], [0, 10000, 0]], dtype=np.float32)
        z_data = np.array([[0, 0, 0], [0, 0, 10000]], dtype=np.float32)
        self._xaxis.set_data(pos=x_data)
        self._yaxis.set_data(pos=y_data)
        self._zaxis.set_data(pos=z_data)

        xtick_pos = np.array([[[i * 100, -50, 0], [i * 100, 50, 0]] for i in range(1, 101)], dtype=np.float32).reshape(
            -1, 3)
        ytick_pos = np.array([[[-50, i * 100, 0], [50, i * 100, 0]] for i in range(1, 101)], dtype=np.float32).reshape(
            -1, 3)
        ztick_pos = np.array([[[0, -50, i * 100], [0, 50, i * 100]] for i in range(1, 101)], dtype=np.float32).reshape(
            -1, 3)

        xtick_connect = np.array([[i * 2, i * 2 + 1] for i in range(0, 100)], dtype=np.uint32).reshape(-1, 2)
        ytick_connect = np.array([[i * 2, i * 2 + 1] for i in range(0, 100)], dtype=np.uint32).reshape(-1, 2)
        ztick_connect = np.array([[i * 2, i * 2 + 1] for i in range(0, 100)], dtype=np.uint32).reshape(-1, 2)

        self._xticks.set_data(pos=xtick_pos, connect=xtick_connect)
        self._yticks.set_data(pos=ytick_pos, connect=ytick_connect)
        self._zticks.set_data(pos=ztick_pos, connect=ztick_connect)


# 바로 밑의 2개 함수는 아직 정리가 더 필요함.

def calculate_angle(point1, point2, point3):
    # Create vectors from the points
    v1 = point1 - point2
    v2 = point3 - point2
    
    # Calculate the dot product of the vectors
    dot_product = np.dot(v1, v2)
    
    # Calculate the magnitudes (lengths) of the vectors
    magnitude_v1 = np.linalg.norm(v1)
    magnitude_v2 = np.linalg.norm(v2)
    
    # Handle the case where the magnitude of either vector is 0
    if magnitude_v1 == 0 or magnitude_v2 == 0:
        return 0.0
    
    # Calculate the angle in radians
    try:
        angle_radian = np.arccos(dot_product / (magnitude_v1 * magnitude_v2))
    except ValueError:
        # In case of numerical issues leading to a domain error in arccos
        return 0.0
    
    # Convert the angle to degrees
    angle_degree = np.degrees(angle_radian)
    
    return angle_degree

def extract_joint_angles(frame_p3ds):
    angles = {}
    # Define key joint indices
    left_shoulder, left_elbow, left_wrist = 0, 2, 4
    right_shoulder, right_elbow, right_wrist = 1,3,5
    left_hip, left_knee, left_ankle = 6, 8, 10
    right_hip, right_knee, right_ankle = 7, 9, 11
    left_toe, right_toe = 14, 15  # Assuming 14, 15 are the indices for left and right toes respectively

    # calculate arm angles
    angles['left_elbow'] = calculate_angle(np.array(frame_p3ds[left_shoulder]), np.array(frame_p3ds[left_elbow]), np.array(frame_p3ds[left_wrist]))
    angles['right_elbow'] = calculate_angle(np.array(frame_p3ds[right_shoulder]), np.array(frame_p3ds[right_elbow]), np.array(frame_p3ds[right_wrist]))

    # Calculate hip angles
    angles['left_hip'] = calculate_angle(np.array(frame_p3ds[left_knee]), np.array(frame_p3ds[left_hip]), np.array(frame_p3ds[right_hip]))
    angles['right_hip'] = calculate_angle(np.array(frame_p3ds[right_knee]), np.array(frame_p3ds[right_hip]), np.array(frame_p3ds[left_hip]))
    
    # Calculate knee angles
    angles['left_knee'] = calculate_angle(np.array(frame_p3ds[left_ankle]), np.array(frame_p3ds[left_knee]), np.array(frame_p3ds[left_hip]))
    angles['right_knee'] = calculate_angle(np.array(frame_p3ds[right_ankle]), np.array(frame_p3ds[right_knee]), np.array(frame_p3ds[right_hip]))
    
    # Calculate ankle angles
    angles['left_ankle'] = calculate_angle(np.array(frame_p3ds[left_toe]), np.array(frame_p3ds[left_ankle]), np.array(frame_p3ds[left_knee]))
    angles['right_ankle'] = calculate_angle(np.array(frame_p3ds[right_toe]), np.array(frame_p3ds[right_ankle]), np.array(frame_p3ds[right_knee]))

    angles['new_angle'] = calculate_angle(np.array(frame_p3ds)[2], np.array(frame_p3ds[0]), np.array(frame_p3ds[6]))
    
    return angles


def _make_homogeneous_rep_matrix(R, t):
    P = np.zeros((4,4))
    P[:3,:3] = R
    P[:3, 3] = t.reshape(3)
    P[3,3] = 1
    return P

#direct linear transform
def DLT(P1, P2, point1, point2):

    A = [point1[1]*P1[2,:] - P1[1,:],
         P1[0,:] - point1[0]*P1[2,:],
         point2[1]*P2[2,:] - P2[1,:],
         P2[0,:] - point2[0]*P2[2,:]
        ]
    A = np.array(A).reshape((4,4))
    #print('A: ')
    #print(A)

    B = A.transpose() @ A
    from scipy import linalg
    U, s, Vh = linalg.svd(B, full_matrices = False)

    #print('Triangulated point: ')
    #print(Vh[3,0:3]/Vh[3,3])
    return Vh[3,0:3]/Vh[3,3]

def read_camera_parameters(camera_id):

    inf = open('camera_parameters/c' + str(camera_id) + '.dat', 'r')

    cmtx = []
    dist = []

    line = inf.readline()
    for _ in range(3):
        line = inf.readline().split()
        line = [float(en) for en in line]
        cmtx.append(line)

    line = inf.readline()
    line = inf.readline().split()
    line = [float(en) for en in line]
    dist.append(line)

    return np.array(cmtx), np.array(dist)

def read_rotation_translation(camera_id, savefolder = 'camera_parameters/'):

    inf = open(savefolder + 'rot_trans_c'+ str(camera_id) + '.dat', 'r')

    inf.readline()
    rot = []
    trans = []
    for _ in range(3):
        line = inf.readline().split()
        line = [float(en) for en in line]
        rot.append(line)

    inf.readline()
    for _ in range(3):
        line = inf.readline().split()
        line = [float(en) for en in line]
        trans.append(line)

    inf.close()
    return np.array(rot), np.array(trans)

def _convert_to_homogeneous(pts):
    pts = np.array(pts)
    if len(pts.shape) > 1:
        w = np.ones((pts.shape[0], 1))
        return np.concatenate([pts, w], axis = 1)
    else:
        return np.concatenate([pts, [1]], axis = 0)

def get_projection_matrix(camera_id):

    #read camera parameters
    cmtx, dist = read_camera_parameters(camera_id)
    rvec, tvec = read_rotation_translation(camera_id)

    #calculate projection matrix
    P = cmtx @ _make_homogeneous_rep_matrix(rvec, tvec)[:3,:]
    return P

def write_keypoints_to_disk(filename, kpts):
    fout = open(filename, 'w')

    for frame_kpts in kpts:
        for kpt in frame_kpts:
            if len(kpt) == 2:
                fout.write(str(kpt[0]) + ' ' + str(kpt[1]) + ' ')
            else:
                fout.write(str(kpt[0]) + ' ' + str(kpt[1]) + ' ' + str(kpt[2]) + ' ')

        fout.write('\n')
    fout.close()

if __name__ == '__main__':

    P2 = get_projection_matrix(0)
    P1 = get_projection_matrix(1)
    
    
def save_2d_pose(frame, results, exclude_indices):
    pose_keypoints = [16, 14, 12, 11, 13, 15, 24, 23, 25, 26, 27, 28, 29, 31, 30, 32]
    """
    Parameters:
        ax (matplotlib.axes.Axes): The axes object where the landmarks will be plotted.
        frame (numpy.ndarray): The video frame from which landmarks are detected.
        results (mediapipe.python.solutions.pose.PoseLandmarks): The pose landmarks detection results.
        exclude_indices (list): Indices of landmarks to exclude from drawing.
        body_parts (dict): Dictionary mapping body parts to indices pairs for connections.
        part_colors (dict): Dictionary mapping body parts to their respective color index.
        colors (list): List of colors to be used for different body parts.
    """
    if results.pose_landmarks:
        H, W = frame.shape[0], frame.shape[1]  # Image height and width
        
        landmarks = results.pose_landmarks.landmark
        
        # Create dictionary of transformed landmarks while filtering by exclude_indices
        transformed_landmarks = {i: (landmark.x * W, H - (landmark.y * H))
                                 for i, landmark in enumerate(landmarks) if i not in exclude_indices}
    
    else:
        transformed_landmarks = {i: (-1, -1) for i in range(len(pose_keypoints))}

    return transformed_landmarks
    
    
def draw_2d_pose(ax, frame, results, exclude_indices, body_parts, part_colors, colors):
    """
    Draw the connections between pose landmarks on a matplotlib axis.
    """
    if results.pose_landmarks:
        H, W = frame.shape[0], frame.shape[1]  # Image height and width
        
        landmarks = results.pose_landmarks.landmark
        
        # Create dictionary of transformed landmarks while filtering by exclude_indices
        transformed_landmarks = {i: (landmark.x * W, H - (landmark.y * H))
                                 for i, landmark in enumerate(landmarks) if i not in exclude_indices}
        
        ax.clear()
        ax.set_xlim([0, W])
        ax.set_ylim([0, H])
        ax.grid(True)  # Optional, based on visualization preferences
        
        # Draw connection lines for each body part
        for part, links in body_parts.items():
            color = colors[part_colors[part]]
            for start_idx, end_idx in links:
                if start_idx in transformed_landmarks and end_idx in transformed_landmarks:
                    start_pt = transformed_landmarks[start_idx]
                    end_pt = transformed_landmarks[end_idx]
                    ax.plot([start_pt[0], end_pt[0]],
                            [start_pt[1], end_pt[1]],
                            color=color, linewidth=2)
