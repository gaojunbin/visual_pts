import plotly.graph_objects as go
import plotly
import torch
import numpy as np

class Visual(object):
    '''
    Note:
        Base class for visialization.
        All specific dataset visialization will be inherited from Visual class.
    Input:
        pts: origin point clouds data with the shape of (N, 3 or 4). 
            N equals to the number of points.
            3 or 4 contains [x,y,z] or [x,y,z,r].
        gt: (Optional) ground truth of bboxes with the shape of (M, 7). 
            M equals to the number of bboxes.
            7 contains [x, y, z, dx, dy, dz, heading],(x,y,z) is the 3d box center.
    Args:
        col: Optional('height'(default) ==> color changes along z axis, 'distance' ==> color changes along x-o-y plane)
        show_level: Optional(0,1,2)
        ...
    '''
    def __init__(self, pts=None, gt=None):
        self.pts_colorscale = 'deep'
        self.pts_size = 1
        self.show_level = 1
        self.col = 'height'
        self.pts = pts
        self.gt = gt
        
    def set_show(self, pts_size=1, pts_colorscale='deep', col='height'):
        self.pts_colorscale = pts_colorscale
        self.pts_size = pts_size
        self.col = col
        
    def give_data(self, pts, gt=None):
        self.pts = pts
        self.gt = gt
        
    def check_numpy_to_torch(slef, x):
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x).float(), True
        return x, False
    
    def rotate_points_along_z(self, points, angle):
        """
        Args:
            points: (B, N, 3 + C)
            angle: (B), angle along z-axis, angle increases x ==> y
        Returns:

        """
        points, is_numpy = self.check_numpy_to_torch(points)
        angle, _ = self.check_numpy_to_torch(angle)

        cosa = torch.cos(angle)
        sina = torch.sin(angle)
        zeros = angle.new_zeros(points.shape[0])
        ones = angle.new_ones(points.shape[0])
        rot_matrix = torch.stack((
            cosa,  sina, zeros,
            -sina, cosa, zeros,
            zeros, zeros, ones
        ), dim=1).view(-1, 3, 3).float()
        points_rot = torch.matmul(points[:, :, 0:3], rot_matrix)
        points_rot = torch.cat((points_rot, points[:, :, 3:]), dim=-1)
        return points_rot.numpy() if is_numpy else points_rot

    def boxes_to_corners_3d(self, boxes3d):
        """
            7 -------- 4
           /|         /|
          6 -------- 5 .
          | |        | |
          . 3 -------- 0
          |/         |/
          2 -------- 1
        Args:
            boxes3d:  (N, 7) [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center

        Returns:
        """
        boxes3d, is_numpy = self.check_numpy_to_torch(boxes3d)

        template = boxes3d.new_tensor((
            [1, 1, -1], [1, -1, -1], [-1, -1, -1], [-1, 1, -1],
            [1, 1, 1], [1, -1, 1], [-1, -1, 1], [-1, 1, 1],
        )) / 2

        corners3d = boxes3d[:, None, 3:6].repeat(1, 8, 1) * template[None, :, :]
        corners3d = self.rotate_points_along_z(corners3d.view(-1, 8, 3), boxes3d[:, 6]).view(-1, 8, 3)
        corners3d += boxes3d[:, None, 0:3]

        return corners3d.numpy() if is_numpy else corners3d

    def draw_corners_3d_boxes(self, corners_3d):
        xs = corners_3d[:,:,0]
        ys = corners_3d[:,:,1]
        zs = corners_3d[:,:,2]

        #the start and end point for each line
        pairs = [(0,1), (1,2), (2,3), (3,0), (0,4), (1,5), (2,6), (3,7), (4,5), (5,6), (6,7), (7,4)]

        corners_3d_points = go.Scatter3d(
            x=xs.reshape(-1),
            y=ys.reshape(-1),
            z=zs.reshape(-1),
            mode='markers',
            name='bbox_markers',
            marker=dict(
                    size=2,
                )
        )

        x_lines = list()
        y_lines = list()
        z_lines = list()

        #create the coordinate list for the lines
        for k in range(len(xs)):
            x = xs[k]
            y = ys[k]
            z = zs[k]
            for p in pairs:
                for i in range(2):
                    x_lines.append(x[p[i]])
                    y_lines.append(y[p[i]])
                    z_lines.append(z[p[i]])
                x_lines.append(None)
                y_lines.append(None)
                z_lines.append(None)

        corners_3d_lines = go.Scatter3d(
            x=x_lines,
            y=y_lines,
            z=z_lines,
            mode='lines',
            name='bboxes'
        )
        return corners_3d_points, corners_3d_lines

    def visualize_pts(self, points, show_intensity=False):
        if not isinstance(points, np.ndarray):
            points = points.cpu().numpy()

        # TODO
        if show_intensity:
            assert points.shape[-1] == 4, "point cloud data here should be 4 dims with x,y,z and r"
        else:
            assert points.shape[-1] in [3,4], "point cloud data here should be 3(x,y,z) or 4(x,y,z,r) dims"
        
        def trans_col(col):
            if col=='height':
                return points[:,2]
            if col=='distance':
                return np.sqrt(points[:,0]**2+points[:,1]**2)
            if col=='intensity':
                return points[:,3]
        
        pts = go.Scatter3d(
                x=points[:,0], y=points[:,1], z=points[:,2], 
                mode='markers',
                name='point clouds',
                marker=dict(
                    size=self.pts_size,
                    color=trans_col(self.col),   # set color to an array/list of desired values
                    colorscale=self.pts_colorscale,   # choose a colorscale
                    opacity=1
                )
            )
        return pts

    def draw_scenes(self, show_level=1, fig_name=None):
        assert show_level in [0,1,2], "show_level should be in [0,1,2],\
        0: only show point clouds.\
        1: show pts and bboxes.(default)\
        2: show pts and bboes with box markers"
        
        layout=dict(
            scene=dict(
                xaxis=dict(visible=True),
                yaxis=dict(visible=True),
                zaxis=dict(visible=True)
            )
        )
        assert self.pts is not None, "there are no point clouds to visiliaze."
        points = self.visualize_pts(self.pts)
        if show_level>0:
            assert self.gt is not None, "there are no gts to visiliaze."
            corners_3d_boxes = self.boxes_to_corners_3d(self.gt)
            corners_3d_points, corners_3d_lines = self.draw_corners_3d_boxes(corners_3d_boxes) 
        
        if show_level == 0:
            data = [points]
        elif show_level == 1:
            data = [points, corners_3d_lines]
        else:
            data = [points, corners_3d_lines, corners_3d_points]

        fig = go.Figure(data=data, layout=dict(
            scene=dict(
                xaxis=dict(visible=True),
                yaxis=dict(visible=True),
                zaxis=dict(visible=True)
            )
        )
                       )
        plotly.offline.iplot(fig, filename= fig_name)

        
class Object3d(object):
    def __init__(self, line):
        label = line.strip().split(' ')
        self.cls_type = label[0]
        self.h = float(label[8])
        self.w = float(label[9])
        self.l = float(label[10])
        self.loc = np.array((float(label[11]), float(label[12]), float(label[13])), dtype=np.float32)
        self.ry = float(label[14])


class Calibration(object):
    def __init__(self, calib_file):
        if not isinstance(calib_file, dict):
            calib = self.get_calib_from_file(calib_file)
        else:
            calib = calib_file

        self.P2 = calib['P2']  # 3 x 4
        self.R0 = calib['R0']  # 3 x 3
        self.V2C = calib['Tr_velo2cam']  # 3 x 4

        # Camera intrinsics and extrinsics
        self.cu = self.P2[0, 2]
        self.cv = self.P2[1, 2]
        self.fu = self.P2[0, 0]
        self.fv = self.P2[1, 1]
        self.tx = self.P2[0, 3] / (-self.fu)
        self.ty = self.P2[1, 3] / (-self.fv)
        
    def get_calib_from_file(self, calib_file):
        with open(calib_file) as f:
            lines = f.readlines()

        obj = lines[2].strip().split(' ')[1:]
        P2 = np.array(obj, dtype=np.float32)
        obj = lines[3].strip().split(' ')[1:]
        P3 = np.array(obj, dtype=np.float32)
        obj = lines[4].strip().split(' ')[1:]
        R0 = np.array(obj, dtype=np.float32)
        obj = lines[5].strip().split(' ')[1:]
        Tr_velo_to_cam = np.array(obj, dtype=np.float32)

        return {'P2': P2.reshape(3, 4),
                'P3': P3.reshape(3, 4),
                'R0': R0.reshape(3, 3),
                'Tr_velo2cam': Tr_velo_to_cam.reshape(3, 4)}

    def cart_to_hom(self, pts):
        """
        :param pts: (N, 3 or 2)
        :return pts_hom: (N, 4 or 3)
        """
        pts_hom = np.hstack((pts, np.ones((pts.shape[0], 1), dtype=np.float32)))
        return pts_hom

    def rect_to_lidar(self, pts_rect):
        """
        :param pts_lidar: (N, 3)
        :return pts_rect: (N, 3)
        """
        pts_rect_hom = self.cart_to_hom(pts_rect)  # (N, 4)
        R0_ext = np.hstack((self.R0, np.zeros((3, 1), dtype=np.float32)))  # (3, 4)
        R0_ext = np.vstack((R0_ext, np.zeros((1, 4), dtype=np.float32)))  # (4, 4)
        R0_ext[3, 3] = 1
        V2C_ext = np.vstack((self.V2C, np.zeros((1, 4), dtype=np.float32)))  # (4, 4)
        V2C_ext[3, 3] = 1

        pts_lidar = np.dot(pts_rect_hom, np.linalg.inv(np.dot(R0_ext, V2C_ext).T))
        return pts_lidar[:, 0:3]

    def lidar_to_rect(self, pts_lidar):
        """
        :param pts_lidar: (N, 3)
        :return pts_rect: (N, 3)
        """
        pts_lidar_hom = self.cart_to_hom(pts_lidar)
        pts_rect = np.dot(pts_lidar_hom, np.dot(self.V2C.T, self.R0.T))
        # pts_rect = reduce(np.dot, (pts_lidar_hom, self.V2C.T, self.R0.T))
        return pts_rect

    def rect_to_img(self, pts_rect):
        """
        :param pts_rect: (N, 3)
        :return pts_img: (N, 2)
        """
        pts_rect_hom = self.cart_to_hom(pts_rect)
        pts_2d_hom = np.dot(pts_rect_hom, self.P2.T)
        pts_img = (pts_2d_hom[:, 0:2].T / pts_rect_hom[:, 2]).T  # (N, 2)
        pts_rect_depth = pts_2d_hom[:, 2] - self.P2.T[3, 2]  # depth in rect camera coord
        return pts_img, pts_rect_depth

    def lidar_to_img(self, pts_lidar):
        """
        :param pts_lidar: (N, 3)
        :return pts_img: (N, 2)
        """
        pts_rect = self.lidar_to_rect(pts_lidar)
        pts_img, pts_depth = self.rect_to_img(pts_rect)
        return pts_img, pts_depth

    def img_to_rect(self, u, v, depth_rect):
        """
        :param u: (N)
        :param v: (N)
        :param depth_rect: (N)
        :return:
        """
        x = ((u - self.cu) * depth_rect) / self.fu + self.tx
        y = ((v - self.cv) * depth_rect) / self.fv + self.ty
        pts_rect = np.concatenate((x.reshape(-1, 1), y.reshape(-1, 1), depth_rect.reshape(-1, 1)), axis=1)
        return pts_rect

    def corners3d_to_img_boxes(self, corners3d):
        """
        :param corners3d: (N, 8, 3) corners in rect coordinate
        :return: boxes: (None, 4) [x1, y1, x2, y2] in rgb coordinate
        :return: boxes_corner: (None, 8) [xi, yi] in rgb coordinate
        """
        sample_num = corners3d.shape[0]
        corners3d_hom = np.concatenate((corners3d, np.ones((sample_num, 8, 1))), axis=2)  # (N, 8, 4)

        img_pts = np.matmul(corners3d_hom, self.P2.T)  # (N, 8, 3)

        x, y = img_pts[:, :, 0] / img_pts[:, :, 2], img_pts[:, :, 1] / img_pts[:, :, 2]
        x1, y1 = np.min(x, axis=1), np.min(y, axis=1)
        x2, y2 = np.max(x, axis=1), np.max(y, axis=1)

        boxes = np.concatenate((x1.reshape(-1, 1), y1.reshape(-1, 1), x2.reshape(-1, 1), y2.reshape(-1, 1)), axis=1)
        boxes_corner = np.concatenate((x.reshape(-1, 8, 1), y.reshape(-1, 8, 1)), axis=2)

        return boxes, boxes_corner

class Kitty_Visual(Visual):
    '''
    Note:
        Specific class for kitty dataset visialization.
        Inherited from Visual class.
    Input:
        data_path: The root path of kitty dataset.
        mode: Optional('training' ==> sample in training loader,'testing' ==> sample in testing loader)
    '''
    
    def __init__(self, data_path, mode='training'):
        super().__init__()
        self.data_path = data_path
        assert mode in ['training','testing'], "mode could only be 'training' or 'testing'"
        self.mode = mode
        
    def draw_scenes(self, index, show_level=1, fig_name=None):
        pts,gt = self.get_data_from_index(index)
        self.give_data(pts, gt)
        super().draw_scenes(show_level, fig_name)
    
    def get_data_from_index(self, index):
        
        pts = self.read_pts_from_index(index)
        gt = self.read_gt_from_index(index)
        
        return pts,gt
    
    def read_pts_from_index(self, index):
        pts_file = self.data_path + '/' + self.mode +'/velodyne/' + ('%s.bin' % index)
        return np.fromfile(pts_file,dtype=np.float32,count=-1).reshape([-1,4])
        
    def read_gt_from_index(self, index):
        obj_list = self.get_label(index)
        calib = self.get_calib(index)

        name = np.array([obj.cls_type for obj in obj_list])
        num_objects = len([obj.cls_type for obj in obj_list if obj.cls_type != 'DontCare'])
        location = np.concatenate([obj.loc.reshape(1, 3) for obj in obj_list], axis=0)
        dimensions = np.array([[obj.l, obj.h, obj.w] for obj in obj_list])  # lhw(camera) format
        rotation_y= np.array([obj.ry for obj in obj_list])
        
        loc = location[:num_objects]
        dims = dimensions[:num_objects]
        rots = rotation_y[:num_objects]
        loc_lidar = calib.rect_to_lidar(loc)
        l, h, w = dims[:, 0:1], dims[:, 1:2], dims[:, 2:3]
        loc_lidar[:, 2] += h[:, 0] / 2
        gt_boxes_lidar = np.concatenate([loc_lidar, l, w, h, -(np.pi / 2 + rots[..., np.newaxis])], axis=1)
        return gt_boxes_lidar
    
    def get_label(self, index):
        label_file = self.data_path + '/' + self.mode + '/label_2/' + ('%s.txt' % index)
        with open(label_file, 'r') as f:
            lines = f.readlines()
        objects = [Object3d(line) for line in lines]
        return objects
    
    def get_calib(self, index):
        calib_file = self.data_path + '/' + self.mode + '/calib/' + ('%s.txt' % index)
        return Calibration(calib_file)


if __name__ == '__main__':
    # example
    # kitty = Kitty_Visual("/home/gaojunbin/project/code/pv_rcnn/data/kitti")
    kitty = Kitty_Visual("../kitti")
    kitty.draw_scenes('000043')


