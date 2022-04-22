from multiprocessing import Process, Queue
from queue import Empty

import os
import numpy as np
import open3d as o3d
import torch

def normalize(x):
    return x / np.linalg.norm(x)

def align_vector_to_another(a=np.array([0, 0, 1]), b=np.array([1, 0, 0])):
    """
    Aligns vector a to vector b with axis angle rotation
    """
    if np.array_equal(a, b):
        return None, None
    axis_ = np.cross(a, b)
    axis_ = axis_ / np.linalg.norm(axis_)
    angle = np.arccos(np.dot(a, b))

    return axis_, angle


def normalized(a, axis=-1, order=2):
    """Normalizes a numpy array of points"""
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis), l2


class LineMesh(object):
    def __init__(self, points, lines=None, colors=[0, 1, 0], radius=0.15):
        # code slip adapted from https://github.com/isl-org/Open3D/pull/738#issuecomment-564785941
        """Creates a line represented as sequence of cylinder triangular meshes

        Arguments:
            points {ndarray} -- Numpy array of ponts Nx3.

        Keyword Arguments:
            lines {list[list] or None} -- List of point index pairs denoting line segments. If None, implicit lines from ordered pairwise points. (default: {None})
            colors {list} -- list of colors, or single color of the line (default: {[0, 1, 0]})
            radius {float} -- radius of cylinder (default: {0.15})
        """
        self.points = np.array(points)
        self.lines = np.array(
            lines) if lines is not None else self.lines_from_ordered_points(self.points)
        self.colors = np.array(colors)
        self.radius = radius
        self.cylinder_segments = []

        self.create_line_mesh()

    @staticmethod
    def lines_from_ordered_points(points):
        lines = [[i, i + 1] for i in range(0, points.shape[0] - 1, 1)]
        return np.array(lines)

    def merge_cylinder_segments(self):

        vertices_list = [np.asarray(mesh.vertices)
                         for mesh in self.cylinder_segments]
        triangles_list = [np.asarray(mesh.triangles)
                          for mesh in self.cylinder_segments]
        triangles_offset = np.cumsum([v.shape[0] for v in vertices_list])
        triangles_offset = np.insert(triangles_offset, 0, 0)[:-1]

        vertices = np.vstack(vertices_list)
        triangles = np.vstack(
            [triangle + offset for triangle, offset in zip(triangles_list, triangles_offset)])

        merged_mesh = o3d.geometry.TriangleMesh(o3d.open3d.utility.Vector3dVector(vertices),
                                                o3d.open3d.utility.Vector3iVector(triangles))
        color = self.colors if self.colors.ndim == 1 else self.colors[0]
        merged_mesh.paint_uniform_color(color)
        self.cylinder_segments = merged_mesh

    def create_line_mesh(self):
        first_points = self.points[self.lines[:, 0], :]
        second_points = self.points[self.lines[:, 1], :]
        line_segments = second_points - first_points
        line_segments_unit, line_lengths = normalized(line_segments)

        z_axis = np.array([0, 0, 1])
        # Create triangular mesh cylinder segments of line
        for i in range(line_segments_unit.shape[0]):
            line_segment = line_segments_unit[i, :]
            line_length = line_lengths[i]
            # get axis angle rotation to allign cylinder with line segment
            axis, angle = align_vector_to_another(z_axis, line_segment)
            # Get translation vector
            translation = first_points[i, :] + line_segment * line_length * 0.5
            # create cylinder and apply transformations
            cylinder_segment = o3d.geometry.TriangleMesh.create_cylinder(
                self.radius, line_length)
            cylinder_segment = cylinder_segment.translate(
                translation, relative=False)
            if axis is not None:
                axis_a = axis * angle
                cylinder_segment = cylinder_segment.rotate(
                    R=o3d.geometry.get_rotation_matrix_from_axis_angle(axis_a), center=cylinder_segment.get_center())
                # cylinder_segment = cylinder_segment.rotate(
                #   axis_a, center=True, type=o3d.geometry.RotationType.AxisAngle)
            # color cylinder
            color = self.colors if self.colors.ndim == 1 else self.colors[i, :]
            cylinder_segment.paint_uniform_color(color)

            self.cylinder_segments.append(cylinder_segment)
        self.merge_cylinder_segments()

    # def add_line(self, vis):
    #     """Adds this line to the visualizer"""
    #     for cylinder in self.cylinder_segments:
    #         vis.add_geometry(cylinder)

    # def remove_line(self, vis):
    #     """Removes this line from the visualizer"""
    #     for cylinder in self.cylinder_segments:
    #         vis.remove_geometry(cylinder)


def create_camera_actor(i, is_gt=False, scale=0.005):
    """ 
    build open3d camera polydata

    """
    cam_points = scale * np.array([
        [0,   0,   0],
        [-1,  -1, 1.5],
        [1,  -1, 1.5],
        [1,   1, 1.5],
        [-1,   1, 1.5],
        [-0.5, 1, 1.5],
        [0.5, 1, 1.5],
        [0, 1.2, 1.5]])

    cam_lines = np.array([[1, 2], [2, 3], [3, 4], [4, 1], [1, 3], [2, 4],
                          [1, 0], [0, 2], [3, 0], [0, 4], [5, 7], [7, 6]])
    points = []
    for cam_line in cam_lines:
        begin_points, end_points = cam_points[cam_line[0]
                                              ], cam_points[cam_line[1]]
        t_vals = np.linspace(0., 1., 100)
        begin_points, end_points
        point = begin_points[None, :] * \
            (1.-t_vals)[:, None] + end_points[None, :] * (t_vals)[:, None]
        points.append(point)
    points = np.concatenate(points)
    color = (0.0, 0.0, 0.0) if is_gt else (1.0, .0, .0)
    camera_actor = o3d.geometry.PointCloud(
        points=o3d.utility.Vector3dVector(points))
    camera_actor.paint_uniform_color(color)
    # line_mesh = LineMesh(cam_points, cam_lines.astype(np.long), color, radius=0.01)
    # camera_actor = line_mesh.cylinder_segments
    # camera_actor = o3d.geometry.LineSet(
    #     points=o3d.utility.Vector3dVector(cam_points),
    #     lines=o3d.utility.Vector2iVector(cam_lines))
    # camera_actor.paint_uniform_color(color)

    return camera_actor


def create_point_cloud_actor(points, colors):
    """
    open3d point cloud from numpy array 

    """

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.colors = o3d.utility.Vector3dVector(colors)

    return point_cloud


def draw_trajectory(queue, output, init_pose, cam_scale, save_rendering, near, estimate_c2w_list, gt_c2w_list, render_every_frame):

    draw_trajectory.queue = queue
    draw_trajectory.cameras = {}
    draw_trajectory.points = {}
    draw_trajectory.ix = 0
    draw_trajectory.warmup = 0
    draw_trajectory.mesh = None
    draw_trajectory.frame_idx = 0
    draw_trajectory.traj_actor = None
    draw_trajectory.traj_actor_gt = None
    draw_trajectory.pose = False
    if save_rendering:
        os.system(f"rm -rf {output}/tmp_rendering")

    def animation_callback(vis):
        cam = vis.get_view_control().convert_to_pinhole_camera_parameters()
        # save_pkl(cam.extrinsic, 'src/tools/demo_cam.pkl')
        draw_trajectory.pose = False
        while True:
            try:
                data = draw_trajectory.queue.get_nowait()
                
                if data[0] == 'pose':
                    i, pose, is_gt = data[1:]
                    draw_trajectory.pose = not is_gt
                    if is_gt:
                        i += 100000
                    # the pose is c2w

                    if i in draw_trajectory.cameras:
                        cam_actor, pose_prev = draw_trajectory.cameras[i]
                        pose_change = pose @ np.linalg.inv(pose_prev)

                        cam_actor.transform(pose_change)
                        vis.update_geometry(cam_actor)

                        if i in draw_trajectory.points:
                            pc = draw_trajectory.points[i]
                            pc.transform(pose_change)
                            vis.update_geometry(pc)

                    else:
                        cam_actor = create_camera_actor(i, is_gt, cam_scale)
                        cam_actor.transform(pose)
                        vis.add_geometry(cam_actor)

                    draw_trajectory.cameras[i] = (cam_actor, pose)
                    if render_every_frame:
                        break
                elif data[0] == 'points':
                    i, points, colors = data[1:]
                    point_actor = create_point_cloud_actor(points, colors)

                    pose = draw_trajectory.cameras[i][1]
                    # point_actor.transform(pose)
                    vis.add_geometry(point_actor)

                    draw_trajectory.points[i] = point_actor

                elif data[0] == 'mesh':
                    meshfile = data[1]
                    if draw_trajectory.mesh is not None:
                        vis.remove_geometry(draw_trajectory.mesh)
                    draw_trajectory.mesh = o3d.io.read_triangle_mesh(meshfile)
                    draw_trajectory.mesh.compute_vertex_normals()
                    new_triangles=np.asarray(draw_trajectory.mesh.triangles)[:, ::-1]
                    draw_trajectory.mesh.triangles=o3d.utility.Vector3iVector(new_triangles)
                    # draw_trajectory.mesh.vertex_normals = o3d.utility.Vector3dVector(
                    #     -np.asarray(draw_trajectory.mesh.vertex_normals))
                    draw_trajectory.mesh.triangle_normals = o3d.utility.Vector3dVector(
                        -np.asarray(draw_trajectory.mesh.triangle_normals))
                    vis.add_geometry(draw_trajectory.mesh)

                elif data[0] == 'traj':
                    i, is_gt = data[1:]
                    
                    # estimate_c2w_list, gt_c2w_list
                    color = (0.0, 0.0, 0.0) if is_gt else (1.0, .0, .0)
                    traj_actor = o3d.geometry.PointCloud(
                        points=o3d.utility.Vector3dVector(gt_c2w_list[1:i, :3, 3] if is_gt else estimate_c2w_list[1:i, :3, 3]))
                    traj_actor.paint_uniform_color(color)
                    # traj_actor.colors = o3d.utility.Vector3dVector(color)
                    
                    # Ts, is_gt = data[1:]
                    # color = (0.0, 0.0, 0.0) if is_gt else (1.0, .0, .0)
                    # traj_actor = o3d.geometry.PointCloud(
                    #     points=o3d.utility.Vector3dVector(Ts))
                    # traj_actor.paint_uniform_color(color)
                    # traj_actor.colors = o3d.utility.Vector3dVector(color)

                    # LineMesh is slow when hanlding a lot of lines
                    # line_mesh = LineMesh(Ts, lines.astype(np.long), color, radius=0.02)
                    # traj_actor = line_mesh.cylinder_segments

                    # traj_actor = o3d.geometry.LineSet(
                    #     points=o3d.utility.Vector3dVector(Ts),
                    #     lines=o3d.utility.Vector2iVector(lines))
                    # traj_actor.paint_uniform_color(color)
                    if is_gt:
                        if draw_trajectory.traj_actor_gt is not None:
                            vis.remove_geometry(draw_trajectory.traj_actor_gt)
                            tmp = draw_trajectory.traj_actor_gt
                            del tmp
                        draw_trajectory.traj_actor_gt = traj_actor
                        vis.add_geometry(draw_trajectory.traj_actor_gt)
                    else:
                        if draw_trajectory.traj_actor is not None:
                            vis.remove_geometry(draw_trajectory.traj_actor)
                            tmp = draw_trajectory.traj_actor
                            del tmp
                        draw_trajectory.traj_actor = traj_actor
                        vis.add_geometry(draw_trajectory.traj_actor)
                        
                    # if draw_trajectory.traj_actor[is_gt] is not None:
                    #     vis.remove_geometry(draw_trajectory.traj_actor[is_gt])
                    #     tmp = draw_trajectory.traj_actor[is_gt]
                    #     del tmp
                    # draw_trajectory.traj_actor[is_gt] = traj_actor
                    # vis.add_geometry(draw_trajectory.traj_actor[is_gt])
                elif data[0] == 'reset':
                    draw_trajectory.warmup = -1

                    for i in draw_trajectory.points:
                        vis.remove_geometry(draw_trajectory.points[i])

                    for i in draw_trajectory.cameras:
                        vis.remove_geometry(draw_trajectory.cameras[i][0])

                    draw_trajectory.cameras = {}
                    draw_trajectory.points = {}

            except Empty:
                break

        # hack to allow interacting with vizualization during inference
        if len(draw_trajectory.cameras) >= draw_trajectory.warmup:
            cam = vis.get_view_control().convert_from_pinhole_camera_parameters(cam)

        vis.poll_events()
        vis.update_renderer()
        if save_rendering:
            if render_every_frame:
                if draw_trajectory.pose:
                    # i, points, colors = data[1:]
                    draw_trajectory.frame_idx += 1
                    os.makedirs(f'{output}/tmp_rendering', exist_ok=True)
                    vis.capture_screen_image(
                        f'{output}/tmp_rendering/{draw_trajectory.frame_idx:06d}.jpg')
            else:       
                # save the renderings, useful when making a video
                draw_trajectory.frame_idx += 1
                os.makedirs(f'{output}/tmp_rendering', exist_ok=True)
                vis.capture_screen_image(
                    f'{output}/tmp_rendering/{draw_trajectory.frame_idx:06d}.jpg')

    vis = o3d.visualization.Visualizer()

    vis.register_animation_callback(animation_callback)
    vis.create_window(window_name=output, height=1080, width=1920)
    # vis.create_window(window_name=output, height=1000, width=1000) #for scannet 00
    vis.get_render_option().line_width = 10.   # sadly no use!
    # vis.get_render_option().point_size = 5 #Replica room1
    vis.get_render_option().point_size = 4 #ScanNet 00
    vis.get_render_option().mesh_show_back_face = False

    ctr = vis.get_view_control()
    ctr.set_constant_z_near(near)
    ctr.set_constant_z_far(1000)
    
    param = ctr.convert_to_pinhole_camera_parameters()
    init_pose[:3, 3] += 2*normalize(init_pose[:3, 2])  # on Apartment OK
    init_pose[:3, 2] *= -1
    init_pose[:3, 1] *= -1
    init_pose = np.linalg.inv(init_pose)

    param.extrinsic = init_pose
    ctr.convert_from_pinhole_camera_parameters(param)

    vis.run()
    vis.destroy_window()


class SLAMFrontend:
    def __init__(self, output, init_pose, cam_scale=1, save_rendering=False, near=4, estimate_c2w_list=None, gt_c2w_list=None, render_every_frame=False):
        self.queue = Queue()
        self.p = Process(target=draw_trajectory, args=(
            self.queue, output, init_pose, cam_scale, save_rendering, near, estimate_c2w_list, gt_c2w_list, render_every_frame))

    def update_pose(self, index, pose, gt=False):
        if isinstance(pose, torch.Tensor):
            pose = pose.cpu().numpy()

        pose[:3, 2] *= -1
        self.queue.put_nowait(('pose', index, pose, gt))

    def update_points(self, index, points, colors):
        if isinstance(points, torch.Tensor):
            points = points.cpu().numpy()
        self.queue.put_nowait(('points', index, points, colors))

    def update_mesh(self, path):
        self.queue.put_nowait(('mesh', path))

    def update_cam_trajectory(self, c2w_list, gt):
        self.queue.put_nowait(('traj', c2w_list, gt))

    def reset(self):
        self.queue.put_nowait(('reset', ))

    def start(self):
        self.p.start()
        return self

    def join(self):
        self.p.join()
