from visual_pts.visual_plotly import Visual
import visual_pts.util_kitti as utils
from matplotlib import pyplot as plt
import os
import cv2
import numpy as np

# TODO: merger with util_kitty
class Object3d(object):
    def __init__(self, line):
        label = line.strip().split(' ')
        self.cls_type = label[0]
        self.h = float(label[8])
        self.w = float(label[9])
        self.l = float(label[10])
        self.loc = np.array((float(label[11]), float(label[12]), float(label[13])), dtype=np.float32)
        self.ry = float(label[14])
        
# TODO: merger with util_kitty
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

class Kitti_Visual_Img(object):
    """Load and parse object data into a usable format."""

    def __init__(self, root_dir, split="training", args=None):
        """root_dir contains training and testing folders"""
        self.root_dir = root_dir
        self.split = split
        print(root_dir, split)
        self.split_dir = os.path.join(root_dir, split)

        if split == "training":
            self.num_samples = 7481
        elif split == "testing":
            self.num_samples = 7518
        else:
            print("Unknown split: %s" % (split))
            exit(-1)

        lidar_dir = "velodyne"
        depth_dir = "depth"
        pred_dir = "pred"
        if args is not None:
            lidar_dir = args.lidar
            depth_dir = args.depthdir
            pred_dir = args.preddir

        self.image_dir = os.path.join(self.split_dir, "image_2")
        self.label_dir = os.path.join(self.split_dir, "label_2")
        self.calib_dir = os.path.join(self.split_dir, "calib")

        self.depthpc_dir = os.path.join(self.split_dir, "depth_pc")
        self.lidar_dir = os.path.join(self.split_dir, lidar_dir)
        self.depth_dir = os.path.join(self.split_dir, depth_dir)
        self.pred_dir = os.path.join(self.split_dir, pred_dir)

    def __len__(self):
        return self.num_samples

    def get_image(self, idx):
        img_filename = os.path.join(self.image_dir, "%s.png" % (idx))
        return utils.load_image(img_filename)

    def get_lidar(self, idx, dtype=np.float32, n_vec=4):
        lidar_filename = os.path.join(self.lidar_dir, "%s.bin" % (idx))
        print(lidar_filename)
        return utils.load_velo_scan(lidar_filename, dtype, n_vec)

    def get_calibration(self, idx):
        calib_filename = os.path.join(self.calib_dir, "%s.txt" % (idx))
        return utils.Calibration(calib_filename)

    def get_label_objects(self, idx):
        label_filename = os.path.join(self.label_dir, "%s.txt" % (idx))
        return utils.read_label(label_filename)

    def get_pred_objects(self, idx):
        pred_filename = os.path.join(self.pred_dir, "%s.txt" % (idx))
        is_exist = os.path.exists(pred_filename)
        if is_exist:
            return utils.read_label(pred_filename)
        else:
            return None

    def get_depth(self, idx):
        img_filename = os.path.join(self.depth_dir, "%s.png" % (idx))
        return utils.load_depth(img_filename)

    def get_depth_image(self, idx):
        img_filename = os.path.join(self.depth_dir, "%s.png" % (idx))
        return utils.load_depth(img_filename)

    def get_depth_pc(self, idx):
        lidar_filename = os.path.join(self.depthpc_dir, "%s.bin" % (idx))
        is_exist = os.path.exists(lidar_filename)
        if is_exist:
            return utils.load_velo_scan(lidar_filename), is_exist
        else:
            return None, is_exist
        # print(lidar_filename, is_exist)
        # return utils.load_velo_scan(lidar_filename), is_exist

    def get_top_down(self, idx):
        pass

    def isexist_pred_objects(self, idx):
        pred_filename = os.path.join(self.pred_dir, "%s.txt" % (idx))
        return os.path.exists(pred_filename)

    def isexist_depth(self, idx):
        depth_filename = os.path.join(self.depth_dir, "%s.txt" % (idx))
        return os.path.exists(depth_filename)
    
    def draw_3Dbbox_in_image(self, data_idx):
        objects = self.get_label_objects(data_idx)
        calib = self.get_calibration(data_idx)
        img = self.get_image(data_idx)
        img_bbox2d, img_bbox3d = self.show_image_with_boxes(img, objects, calib)
        return img_bbox2d, img_bbox3d

    def show_image_with_boxes(self, img, objects, calib, show3d=True, depth=None):
        """ Show image with 2D bounding boxes """
        img1 = np.copy(img)  # for 2d bbox
        img2 = np.copy(img)  # for 3d bbox
        #img3 = np.copy(img)  # for 3d bbox
        #TODO: change the color of boxes
        for obj in objects:
            if obj.type == "DontCare":
                continue
            if obj.type == "Car":
                cv2.rectangle(
                img1,
                (int(obj.xmin), int(obj.ymin)),
                (int(obj.xmax), int(obj.ymax)),
                (0, 255, 0),
                2,
            )
            if obj.type == "Pedestrian":
                cv2.rectangle(
                img1,
                (int(obj.xmin), int(obj.ymin)),
                (int(obj.xmax), int(obj.ymax)),
                (255, 255, 0),
                2,
            )
            if obj.type == "Cyclist":
                cv2.rectangle(
                img1,
                (int(obj.xmin), int(obj.ymin)),
                (int(obj.xmax), int(obj.ymax)),
                (0, 255, 255),
                2,
            )
            box3d_pts_2d, _ = utils.compute_box_3d(obj, calib.P)
            if obj.type == "Car":
                img2 = utils.draw_projected_box3d(img2, box3d_pts_2d, color=(0, 255, 0))
            elif obj.type == "Pedestrian":
                img2 = utils.draw_projected_box3d(img2, box3d_pts_2d, color=(255, 255, 0))
            elif obj.type == "Cyclist":
                img2 = utils.draw_projected_box3d(img2, box3d_pts_2d, color=(0, 255, 255))

        return img1, img2

class Kitti_Visual(Visual):
    '''
    Note:
        Specific class for Kitti dataset visialization.
        Inherited from Visual class.
    Input:
        data_path: The root path of Kitti dataset.
        mode: Optional('training' ==> sample in training loader,'testing' ==> sample in testing loader)
    '''
    
    def __init__(self, data_path, mode='training'):
        super().__init__()
        self.data_path = data_path
        assert mode in ['training','testing'], "mode could only be 'training' or 'testing'"
        self.mode = mode
        self.kiiti_visual_img = Kitti_Visual_Img(data_path, mode)

    def draw_imgs(self, index, img_level):
        img2d, img3d = self.kiiti_visual_img.draw_3Dbbox_in_image(index)
        img2d = img2d[:,:,::-1] 	# transform image to rgb
        img3d = img3d[:,:,::-1]

        if img_level==1:
            plt.imshow(img2d)
            plt.show()
            # cv2.imshow("2dbox", img2d)
        if img_level==2:
            plt.imshow(img3d)
            plt.show()
            # cv2.imshow("3dbox", img3d)
        if img_level==3:
            plt.imshow(img2d)
            plt.show()
            plt.imshow(img3d)
            plt.show()
            # cv2.imshow("2dbox", img2d)
            # cv2.imshow("3dbox", img3d)
            
        return img2d, img3d
        
    def draw_scenes(self, index, show_level=1, img_level=0, fig_name=None):
        pts,gt = self.get_data_from_index(index)
        self.give_data(pts, gt)
        self.draw_imgs(index, img_level)
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
