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
        col: Optional('height'(default) ==> color changes along z axis, 'distance' ==> color changes along x-o-y plane, 'intensity' ==> color changes along intensity)
        show_level: Optional(0,1,2)
        ...
    '''
    def __init__(self, pts=None, gt=None):
        self.pts_colorscale = 'deep'
        self.pts_size = 1
        self.show_level = 1
        self.col = 'height'
        self.axis_range = 100
        self.pts = pts
        self.gt = gt
        
    def set_show(self, pts_size=1, pts_colorscale='deep', col='height',axis_range=100):
        self.pts_colorscale = pts_colorscale
        self.pts_size = pts_size
        self.col = col
        self.axis_range = axis_range
        
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

        fig = go.Figure(data=data, 
            layout=dict(
                scene=dict(
                    xaxis=dict(visible=True),
                    yaxis=dict(visible=True),
                    zaxis=dict(visible=True)
                )
            )
        )
        fig.update_layout(
            scene=dict(
                xaxis=dict(tickmode='auto',nticks=10, range=[-self.axis_range, self.axis_range],autorange=False,),
                yaxis=dict(tickmode='auto',nticks=10, range=[-self.axis_range, self.axis_range],autorange=False,),
                zaxis=dict(tickmode='auto',nticks=10, range=[-self.axis_range, self.axis_range],autorange=False,),
                aspectratio=dict(x=1, y=1, z=1)#改变画布空间比例为1：1：1
                ),
            margin=dict(r=0, l=0, b=0, t=0))

        plotly.offline.iplot(fig, filename= fig_name)
