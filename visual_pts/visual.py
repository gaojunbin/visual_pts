import warnings
import plotly.graph_objects as go
import plotly
import torch
import numpy as np
import random

class Visual(object):
    '''
    Note:
        Base class for visialization.
        All specific dataset visialization will be inherited from Visual class.
    Input:
        pts: origin point clouds data with the shape of (N, 3 or 4). 
            N equals to the number of points.
            3 or 4 contains [x,y,z] or [x,y,z,r].
        gt: (Optional) ground truth of bboxes with the shape of (M, 7 or 8). 
            M equals to the number of bboxes.
            7 or 8 contains [x, y, z, dx, dy, dz, heading] or [x, y, z, dx, dy, dz, heading, class_name],(x,y,z) is the 3d box center.
        label: (Optional) ground truth label of bboxes with the shape of (M, 1).
    '''
    def __init__(self, pts=None, gt=None, label=None):
        self.init_setting()
        self.pts = pts
        self.gt = gt
        self.label = label

    def init_setting(self):
        self.settings = {
            "pts_colorscale": "point clouds color style",
            "pts_size": "point size",
            "col": "Optional('height'(default) ==> color changes along z axis, 'distance' ==> color changes along x-o-y plane, 'intensity' ==> color changes along intensity)",
            "axis_range": "x-y-z axis range, unit: m",
            "category_list": "type: List(str), categories we are interested, None represents interested in all categories",
            "category_map_color": "type: dict(category:color[rgb]), assign each category we are interested a color, None represents interested in all categories and ramdom set colors for each category"
        }
        self.pts_colorscale = 'deep'
        self.pts_size = 0.5
        self.col = 'height'
        self.axis_range = 100
        self.category_list = list()
        self.category_map_color = dict()

    def help_setting(self):
        for (k,v) in self.settings.items():
            print('{}: {}'.format(k,v))

    def set_show(self, **kwargs):
        if kwargs is not None:
            for (k,v) in kwargs.items():
                if k in self.settings.keys():
                    setattr(self,k,v)
        self.check_setting_is_right()

    def check_setting_is_right(self):
        if self.pts_size<=0 or self.pts_size>10:
            assert False,"pts_size only could be in (0,10]"
        if self.col not in ['height','distance','intensity']:
            assert False,"col only could be in ['height','distance','intensity']"
        if self.axis_range<1 or self.axis_range>200:
            assert False,"axis_range only could be in [1,200]"
        if self.category_list:
            if not isinstance(self.category_list,list):
                assert False,"category_list should be a list"
        if self.category_map_color:
            if not isinstance(self.category_map_color,dict):
                assert False,"category_map_color should be a dict"
        if self.category_map_color:
            if self.category_list:
                self.category_list=list()
                print("category_map_color has a higher priority than category_list, Note that category_list has been cleaned!")
                
    def give_data(self, pts, gt=None, label=None):
        self.pts = self.check_torch_list_to_numpy(pts)
        if gt is not None:
            self.gt = self.check_list_to_numpy(gt)
        if label is not None:
            self.label = self.check_numpy_to_list(label)
      
    def check_numpy_to_torch(slef, x):
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x).float(), True
        return x, False
    
    def check_torch_list_to_numpy(self,x):
        if isinstance(x, torch.Tensor):
            return x.cpu().numpy()
        if isinstance(x, list):
            return np.array(x)
        return x

    def check_list_to_numpy(self,x):
        if isinstance(x, list):
            return np.array(x)
        return x
    
    def check_numpy_to_list(self,x):
        if isinstance(x, np.ndarray):
            return x.tolist()
        return x
    
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
            (N,8,3)
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
    
    def map_category_to_colors(self, category_list):
        category_to_colors = dict()
        for category in category_list:
            category_to_colors.update({category: "rgb({},{},{})".format(random.randint(0,255),random.randint(0,255),random.randint(0,255))})
        return category_to_colors

    def give_each_bbox_color_via_category(self, bbox_name, category_to_colors):
        '''
        Args: 
            bbox_name: [N, 1] class name each bbox
            category_to_colors: dict(class_name: color)
        '''
        bbox_map_color = list()
        for category in bbox_name:
            if category in category_to_colors.keys():
                bbox_map_color.append(category_to_colors[category])
            else:
                bbox_map_color.append(None)
        return bbox_map_color

    def draw_corners_3d_boxes(self, corners_3d, name, bbox_map_color):
        if bbox_map_color:
            assert len(name)==corners_3d.shape[0], "the number of bboxes and labels not equal"

        xs = corners_3d[:,:,0]
        ys = corners_3d[:,:,1]
        zs = corners_3d[:,:,2]

        #the start and end point for each line
        pairs = [(0,1), (1,2), (2,3), (3,0), (0,4), (1,5), (2,6), (3,7), (4,5), (5,6), (6,7), (7,4)]

        # corners_3d_points = go.Scatter3d(
        #     x=xs.reshape(-1),
        #     y=ys.reshape(-1),
        #     z=zs.reshape(-1),
        #     mode='markers',
        #     name='bbox_markers',
        #     marker=dict(
        #             size=2,
        #         )
        # )

        corners_3d_lines = []
        #create the coordinate list for the lines
        for k in range(len(name)):
            # ignore uninterested class
            if self.category_list and (name[k] not in self.category_list):
                continue
            if self.category_map_color and (name[k] not in self.category_map_color.keys()):
                continue

            x = xs[k]
            y = ys[k]
            z = zs[k]

            x_lines = list()
            y_lines = list()
            z_lines = list()

            for p in pairs:
                for i in range(2):
                    x_lines.append(x[p[i]])
                    y_lines.append(y[p[i]])
                    z_lines.append(z[p[i]])
                x_lines.append(None)
                y_lines.append(None)
                z_lines.append(None)

            corners_3d_line = go.Scatter3d(
                x=x_lines,
                y=y_lines,
                z=z_lines,
                mode='lines',
                name=name[k],
                marker=dict(
                        color = bbox_map_color[k] if bbox_map_color else 'rgb(255,0,0)',
                    )
            )
            corners_3d_lines.append(corners_3d_line)
        return corners_3d_lines

    def visualize_pts(self, points):
        assert points.shape[-1] in [3,4], "point cloud data here should be 3(x,y,z) or 4(x,y,z,r) dims"
        if not isinstance(points, np.ndarray):
            if isinstance(points, torch.Tensor):
                points = points.cpu().numpy()
            elif isinstance(points, list):
                points = np.array(points)
            else:
                raise TypeError

        def trans_col(col):
            if col=='height':
                return points[:,2]
            if col=='distance':
                return np.sqrt(points[:,0]**2+points[:,1]**2)
            if col=='intensity':
                assert points.shape[-1] == 4, "point cloud data here should be 4 dims with x,y,z and r"
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

    def draw_scenes(self, fig_name=None):
        assert self.pts is not None, "there are no point clouds to visiliaze."
        points = self.visualize_pts(self.pts)
        data = [points]

        if self.gt is not None:
            corners_3d_boxes = self.boxes_to_corners_3d(self.gt[:,:7])
            name = ['object']* len(corners_3d_boxes)
            bbox_map_color = list()
            if (self.gt.shape[1]==8 and self.label is None) or (self.gt.shape[1]==7 and self.label is not None):
                if self.label is None:
                    name = self.check_numpy_to_list(self.gt[:,7])
                else:
                    name = self.label
                if self.category_map_color:
                    bbox_map_color = self.give_each_bbox_color_via_category(name, self.category_map_color)
                else:
                    category_list = list()
                    for each_name in name:
                        if each_name not in category_list:
                            category_list.append(each_name)
                    category_map_color = self.map_category_to_colors(category_list)
                    bbox_map_color = self.give_each_bbox_color_via_category(name, category_map_color)
                        

            corners_3d_lines = self.draw_corners_3d_boxes(corners_3d_boxes, name, bbox_map_color) 

            for corners_3d_line in corners_3d_lines:
                data.append(corners_3d_line)

        # TODO: Optimized to auto zoom canvas ratio
        layout=dict(
            scene=dict(
                xaxis=dict(visible=True),
                yaxis=dict(visible=True),
                zaxis=dict(visible=True)
            )
        )
        fig = go.Figure(
            data=data, 
            layout=layout
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
