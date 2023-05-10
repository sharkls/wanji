import os
import cv2, pickle,glob
import numpy as np
from tqdm import tqdm
from .tiles2lonlat import TilesSystem

class Tile_Map(object):
    def __init__(self, Config, mosaic_tiles, camera_serial, opt):
        self.Longitude = Config[0][0]   # 基站经度
        self.Latitude = Config[0][1]    # 基站纬度
        self.AngleNorth = Config[0][2]  # 基站北向夹角
        self.levelOfDetail = Config[0][3]  # 瓦片图的划分等级，提供的瓦片图级别为23
        self.Cam_points = Config[camera_serial + 1][0]  # 相机特征点坐标
        self.Map_points = Config[camera_serial + 1][1]  # 瓦片图特征点坐标
        self.station_name = Config[4]   # 基站号
        self.Id_list = mosaic_tiles[0]
        self.X_list = mosaic_tiles[1]
        self.Y_list = mosaic_tiles[2]
        self.Use = opt.use
        self.IMAGE_SIZE = 256   # 瓦片图256像素
        self.repeat_pix = {}
        self.M = []             # 透视变换矩阵
        self.pc_xylonlat_tmp = np.zeros((1920, 1080, 4))  # 初始tmp图
        self.cam_pix, self.lon_lat, self.tilesX_tilesY, self.map_pix, self.xyz_list, self.pc_xylonlat_list = [], [], [], [], [], []
        self.pix_lon_lat = {"map_pix": self.map_pix, "cam_pix": self.cam_pix, "lon_lat": self.lon_lat,
                            "tilesX_tilesY": self.tilesX_tilesY,
                            "xyz": self.xyz_list, "pc_xylonlat": self.pc_xylonlat_list}
        self.pc_xylonlat_save = {"pc_xy_lonlat": self.pc_xylonlat_tmp}
        self.camera_serial = str(camera_serial)
        self.camera_type = opt.camera_type[camera_serial]
        self.Dev_name = opt.Dir_path

        # 生成最新的result*文件夹
        self.save_path = opt.save_path + 'result' + str(len(glob.glob(opt.save_path + 'result*'))-1) + '/' + self.camera_type + self.camera_serial + '/'
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        self.test = TilesSystem()


    def plt_cam(self):
        # 读取相机图像
        try:
            img = cv2.imread("../data/camera_img/" + self.Dev_name.split('/')[-1][:-4] + '/Station_' + str(self.station_name) + '/' + str(self.camera_type) + "_" + str(self.camera_serial) + ".bmp")
        except:
            img = cv2.imread("../data/camera_img/" + self.Dev_name.split('/')[-1][:-4] + '/Station_' + str(self.station_name) + '/' + str(self.camera_type) + "_" + str(self.camera_serial) + ".jpg")
        # 遍历所保留的瓦片图
        for Id in self.Id_list:
            # 根据瓦片图的序号求对应的绝对像素坐标
            pixelX_f, pixelY_f = self.test.TileXYToPixelXY(float(Id[0]), float(Id[1]))
            # 根据绝对像素坐标求对应的经纬度坐标
            Lon, Lat = self.test.PixelXYToLatLong(pixelX_f, pixelY_f, self.levelOfDetail)
            # 根据经纬度坐标求xyz坐标
            xyz = self.test.lon_lat_toxy(Lon, Lat, self.Longitude, self.Latitude, self.AngleNorth)
            xyz = xyz[0]
            x_ = round(xyz[0], 1)
            y_ = round(xyz[1], 1)

            x_id = int(Id[0] - (self.X_list[0] - 1))  # -1 表示防止点在画板上越界
            y_id = int(Id[1] - (self.Y_list[0] - 1))  # -1 表示防止点在画板上越界
            xmin, ymin = x_id * self.IMAGE_SIZE, y_id * self.IMAGE_SIZE
            xmax, ymax = xmin + 256, ymin + 256
            # 计算瓦片图的角点像素坐标
            list = [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]]
            list1 = []

            # 遍历瓦片图四个角点坐标，计算映射到相机图像上的像素坐标
            for i, j in list:
                point_tmp = np.array([[i, j]], dtype="float32")
                point = self.test.reCalculateBBS(point_tmp, self.M)
                list1.append([int(point[0][0]), int(point[0][1])])

            # 角点像素超出相机图像范围，则跳过该瓦片图
            if list1[2][0] <= 0 and list1[0][0] >= 1920 and list1[2][1] <= 0 and list1[0][1] >= 1080:
                continue

            # 根据像素上四个角点画线
            for i in range(len(list1)):
                if i < 3:
                    cv2.line(img, (list1[i][0], list1[i][1]), (list1[i + 1][0], list1[i + 1][1]), (0, 255, 0), 1)
                else:
                    cv2.line(img, (list1[i][0], list1[i][1]), (list1[0][0], list1[0][1]), (0, 255, 0), 1)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img, '{}'.format(str(x_)), (
                int((list1[2][0] - list1[0][0]) / 2 + list1[0][0]),
                int((list1[2][1] - list1[0][1]) / 2 + list1[0][1])),
                        font, 0.5, (0, 0, 255), 1)

        # 将瓦片图映射到相机图像，并将瓦片图绘制在相机图像上
        cv2.imwrite(self.save_path + 'Make_cam1.png', img)


    def Tile_icon(self):
        """
            函数功能：根据瓦片图和相机图像上的四个特征点，计算并返回透视变换矩阵M
        """
        # 读取瓦片图
        img = cv2.imread(self.save_path.split(self.camera_type + self.camera_serial)[0] + "Make_tile1.png", 0)
        w, h = 1920, 1080

        # 透视变换矩阵1
        point1 = np.array(self.Map_points, dtype="float32")
        point2 = np.array(self.Cam_points, dtype="float32")
        self.M = cv2.getPerspectiveTransform(point1, point2)  # 透射变换矩阵

        # # 透视变换矩阵2
        if self.Use == 'two':
            point1 = np.array(self.Map_points0, dtype="float32")
            point2 = np.array(self.Cam_points0, dtype="float32")
            self.M0 = cv2.getPerspectiveTransform(point1, point2)

        # 输出透视变化后的拼接完的瓦片图
        out_img = cv2.warpPerspective(img, self.M, (w, h))
        targetPath = os.path.join(self.save_path + "Transform_later.png")
        cv2.imwrite(targetPath, out_img)

        return self.M

    def Tile_icon1(self):
        img = cv2.imread(self.save_path.split(self.camera_type + self.camera_serial)[0] + "Make_tile1.png", 0)
        w, h = 1920, 1080
        point1 = np.array(self.Map_points, dtype="float32")
        point2 = np.array(self.Cam_points, dtype="float32")
        self.M = cv2.findHomography(point1, point2, cv2.RANSAC)
        self.M = np.array(self.M[0], dtype=np.float32)
        out_img = cv2.warpPerspective(img, self.M, (w, h))
        targetPath = os.path.join(self.save_path + "Transform_later1.png")
        cv2.imwrite(targetPath, out_img)
        return self.M

    def image_compose(self):
        """
            函数功能： 将瓦片图投影至相机视野中，并保存相机视野中每个像素点对应的经纬度值
        """

        # 读取瓦片图的掩码图
        mask_tile = cv2.imread('../data/mask/' + self.Dev_name.split('/')[-1][0:-4] + '/mask_tile.png', 0)

        # 读取短焦相机的掩码图
        mask_short = cv2.imread('../data/mask/' + self.Dev_name.split('/')[-1][0:-4] + '/Station_' + str(self.station_name) + '/mask' + '_' + self.camera_serial + '.bmp', 0)

        # 遍历保存的瓦片图
        for Id in tqdm(self.Id_list):
            tilesX = Id[0]
            tilesY = Id[1]

            # 判断瓦片图四个角映射后是否在图像上
            x_id = int(tilesX - (self.X_list[0] - 1))  # -1 表示防止点在画板上越界
            y_id = int(tilesY - (self.Y_list[0] - 1))  # -1 表示防止点在画板上越界
            xmin, ymin = x_id * self.IMAGE_SIZE, y_id * self.IMAGE_SIZE
            xmax, ymax = xmin + 256, ymin + 256
            # 计算瓦片图的角点像素坐标
            list = [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]]
            flag = 0
            for i in list:
                # 计算角点映射后的坐标
                p_x = i[0]
                p_y = i[1]
                point_tmp_ = np.array([[int(p_x), int(p_y)]], dtype="float32")
                point_new = self.test.reCalculateBBS(point_tmp_, self.M)  # 映射后的点
                # print((point_new[0][1]), (point_new[0][0]))
                # 超出相机图像的瓦片图
                if point_new[0][0] <= 0 or point_new[0][1] <= 0 or point_new[0][0] >= 1920 or point_new[0][1] >= 1080:
                    flag += 1
                # 去除鱼眼中黑色无效区域对应的瓦片图
                elif (point_new[0][0] <= 400 or point_new[0][0] >= 1500) and self.camera_type == 'fisheye':
                    flag += 1
                # 去除短焦中非车道部分的瓦片图
                elif self.camera_type == 'short' and mask_short[int(point_new[0][1])][int(point_new[0][0])] == 255:
                    flag += 1
            if flag >= 4:
                continue

            # 遍历瓦片图中每个像素点
            for pix_i in range(1, 257):
                for pix_j in range(1, 257):

                    # 计算瓦片图中每个效果点对应的瓦片图序号求对应的绝对像素坐标
                    pixelX_f, pixelY_f = self.test.TileXYToPixelXY(float(tilesX) + pix_i / 256, float(tilesY) + pix_j / 256)
                    # 根据绝对像素坐标求对应的经纬度坐标
                    Lon, Lat = self.test.PixelXYToLatLong(pixelX_f, pixelY_f, 23)

                    # 计算该瓦片图的序号
                    x_id = int(tilesX - (self.X_list[0] - 1))  # -1 表示防止点在画板上越界
                    y_id = int(tilesY - (self.Y_list[0] - 1))  # -1 表示防止点在画板上越界

                    # 计算瓦片图中点对应在整个瓦片图上的像素坐标,并映射到相机图像上
                    p_x = x_id * self.IMAGE_SIZE + pix_i
                    p_y = y_id * self.IMAGE_SIZE + pix_j

                    # 如果瓦片图的掩码图为白色像素，则跳过（保留车道的瓦片图像素）
                    if mask_tile[p_y, p_x] == 255:
                        continue

                    point_tmp_ = np.array([[int(p_x), int(p_y)]], dtype="float32")
                    point_new = self.test.reCalculateBBS(point_tmp_, self.M)  # 映射后的点

                    # 使用两个透视变换矩阵
                    if self.Use == 'two':
                        if (point_new[0][1] > self.Cam_points.tolist()[0][1]) and (point_new[0][1] > self.Cam_points.tolist()[1][1]):
                            point_tmp_ = np.array([[int(p_x), int(p_y)]], dtype="float32")
                            point_new = self.test.reCalculateBBS(point_tmp_, self.M0)  # 映射后的点

                    # 舍去映射后不再相机图像上的点
                    if point_new[0][0] <= 0 or point_new[0][1] <= 0 or point_new[0][0] >= 1920 or point_new[0][1] >= 1080:
                        continue

                    # if [point_new[0][0],point_new[0][1]] not in self.cam_pix:
                    #     self.repeat_pix[tuple([point_new[0][0],point_new[0][1]])]=0
                    # else:
                    #     self.repeat_pix[tuple([point_new[0][0],point_new[0][1]])]+=1

                    # 瓦片图中像素点映射到相机图像上后，存在多次映射，对映射次数进行统计
                    # if [point_new[0][0], point_new[0][1]] in self.cam_pix:
                    #     self.repeat_pix[point_new[0][0], point_new[0][1]]['cnt'] += 1
                    # else:
                    #     self.repeat_pix[point_new[0][0], point_new[0][1]] = {}
                    #     self.repeat_pix[point_new[0][0], point_new[0][1]]['cnt'] = 1

                    # 将瓦片图像素和经纬度一一对应保存
                    self.map_pix.append([p_x, p_y])                             # 瓦片图上的像素点
                    self.cam_pix.append([point_new[0][0], point_new[0][1]])   # 映射到相机上的像素点
                    self.lon_lat.append([Lon, Lat])                             # 经纬度坐标
                    # self.tilesX_tilesY.append([tilesX, tilesY])

                    # 经纬度->单基站xyz   高度默认值为-4.2
                    xyz = self.test.lon_lat_toxy(Lon, Lat, self.Longitude, self.Latitude, self.AngleNorth)
                    xyz = xyz[0]
                    self.xyz_list.append(xyz)
                    self.pc_xylonlat_list.append(np.array([xyz[0], xyz[1], Lon, Lat]))   # X, Y，经度，维度

        # 按照单基站相机感受野的每个像素顺序存储对应的xy（基站坐标系下的xy）和经纬度
        for i, j in enumerate(self.pix_lon_lat['cam_pix']):
            pix_x = int(j[0])
            pix_y = int(j[1])
            self.pc_xylonlat_tmp[pix_x - 1][pix_y - 1] = self.pc_xylonlat_list[i]


    def Save_pkl(self):
        """
            函数功能：保存pkl文件
        """
        f_save = open(self.save_path + str(self.camera_type) + str(self.camera_serial) + '.pkl', 'wb')
        pickle.dump(self.pc_xylonlat_save, f_save)
        f_save.close()
        # pkl to npy
        self.pkl_to_npy(self.save_path + str(self.camera_type) + str(self.camera_serial) + '.pkl')
        print("Done", self.camera_serial)

    def Read_pkl(self):
        # # 读取
        f_read = open(self.save_path + str(self.camera_type) + str(self.camera_serial)+'.pkl', 'rb')
        dict2 = pickle.load(f_read)
        print(dict2)
        f_read.close()

    def Plt_pkl(self, XXX, pointx, pointy, index_path):
        """
            函数功能：可视化生成的PKL文件（有值显示该像素点，值为[0,0,0,0]时不显示该像素点）
            Input:  'XXX' :--0 --以5m X 5m的颜色块对标定结果进行可视化; --1 --将图像中的xy空值输出到图像中 
                    index_path : --1 --可视化没进行填充的标定结果；--2 --可视化填充后的标定结果
        """
        # 读取标定结果
        if index_path == 1:
            f_read = open(self.save_path + str(self.camera_type) + str(self.camera_serial) + '.pkl', 'rb')
        elif index_path == 2:
            f_read = open(self.save_path + 'padding_' + str(self.camera_type) + str(self.camera_serial) + '.pkl', 'rb')
        dict = pickle.load(f_read)
        f_read.close()

        # 读取对应相机视野图
        img = cv2.imread("../data/camera_img/" + self.Dev_name.split('/')[-1][:-4] + '/Station_' + str(self.station_name) + '/' + str(self.camera_type) + "_" + str(self.camera_serial) + ".bmp")
        w, h = img.shape[1], img.shape[0]
        pc_xy_lonlat = dict['pc_xy_lonlat']

        # 选用不同的可视化模式来进行标定结果可视化
        if XXX == '2':
            for j in range(pointx, pointx+5, 1):
                for i in range(pointy, pointy+5, 1):
                    print(pc_xy_lonlat[i][j])
        else:
            for j in range(h):
                for i in range(w):
                    x = pc_xy_lonlat[i][j][0]
                    y = pc_xy_lonlat[i][j][1]
                    if XXX == '0':
                        if x == 0 and y == 0:
                            continue
                        if 0 < abs(x) % 25 <= 5:
                            if 0 < abs(y) % 25 <= 5:
                                cv2.circle(img, (i + 1, j + 1), 1, (255, 0, 0), thickness=1)
                            elif 5 < abs(y) % 25 <= 10:
                                cv2.circle(img, (i + 1, j + 1), 1, (0, 255, 0), thickness=1)
                            elif 10 < abs(y) % 25 <= 15:
                                cv2.circle(img, (i + 1, j + 1), 1, (0, 0, 255), thickness=1)
                            elif 15 < abs(y) % 25 <= 20:
                                cv2.circle(img, (i + 1, j + 1), 1, (255, 255, 0), thickness=1)
                            elif 20 < abs(y) % 25 <= 25:
                                cv2.circle(img, (i + 1, j + 1), 1, (0, 255, 255), thickness=1)
                        elif 5 < abs(x) % 25 <= 10:
                            if 0 < abs(y) % 25 <= 5:
                                cv2.circle(img, (i + 1, j + 1), 1, (156, 102, 31), thickness=1)
                            elif 5 < abs(y) % 25 <= 10:
                                cv2.circle(img, (i + 1, j + 1), 1, (176, 23, 64), thickness=1)
                            elif 10 < abs(y) % 25 <= 15:
                                cv2.circle(img, (i + 1, j + 1), 1, (237, 145, 33), thickness=1)
                            elif 15 < abs(y) % 25 <= 20:
                                cv2.circle(img, (i + 1, j + 1), 1, (160, 32, 240), thickness=1)
                            elif 20 < abs(y) % 25 <= 25:
                                cv2.circle(img, (i + 1, j + 1), 1, (3, 168, 158), thickness=1)
                        elif 10 < abs(x) % 25 <= 15:
                            if 0 < abs(y) % 25 <= 5:
                                cv2.circle(img, (i + 1, j + 1), 1, (0, 199, 140), thickness=1)
                            elif 5 < abs(y) % 25 <= 10:
                                cv2.circle(img, (i + 1, j + 1), 1, (153, 51, 250), thickness=1)
                            elif 10 < abs(y) % 25 <= 15:
                                cv2.circle(img, (i + 1, j + 1), 1, (199, 97, 20), thickness=1)
                            elif 15 < abs(y) % 25 <= 20:
                                cv2.circle(img, (i + 1, j + 1), 1, (127, 255, 0), thickness=1)
                            elif 20 < abs(y) % 25 <= 25:
                                cv2.circle(img, (i + 1, j + 1), 1, (116, 0, 0), thickness=1)
                        elif 15 < abs(x) % 25 <= 20:
                            if 0 < abs(y) % 25 <= 5:
                                cv2.circle(img, (i + 1, j + 1), 1, (255, 0, 255), thickness=1)
                            elif 5 < abs(y) % 25 <= 10:
                                cv2.circle(img, (i + 1, j + 1), 1, (192, 192, 192), thickness=1)
                            elif 10 < abs(y) % 25 <= 15:
                                cv2.circle(img, (i + 1, j + 1), 1, (255, 127, 80), thickness=1)
                            elif 15 < abs(y) % 25 <= 20:
                                cv2.circle(img, (i + 1, j + 1), 1, (8, 46, 84), thickness=1)
                            elif 20 < abs(y) % 25 <= 25:
                                cv2.circle(img, (i + 1, j + 1), 1, (188, 143, 143), thickness=1)
                        elif 20 < abs(x) % 25 <= 25:
                            if 0 < abs(y) % 25 <= 5:
                                cv2.circle(img, (i + 1, j + 1), 1, (107, 142, 35), thickness=1)
                            elif 5 < abs(y) % 25 <= 10:
                                cv2.circle(img, (i + 1, j + 1), 1, (160, 82, 45), thickness=1)
                            elif 10 < abs(y) % 25 <= 15:
                                cv2.circle(img, (i + 1, j + 1), 1, (3, 168, 158), thickness=1)
                            elif 15 < abs(y) % 25 <= 20:
                                cv2.circle(img, (i + 1, j + 1), 1, (227, 207, 87), thickness=1)
                            elif 20 < abs(y) % 25 <= 25:
                                cv2.circle(img, (i + 1, j + 1), 1, (116, 0, 0), thickness=1)
                        
                    elif XXX == '1':
                        if x == 0 and y == 0:
                            cv2.circle(img, (i + 1, j + 1), 1, (255, 255, 255), thickness=1)
                        else:
                            continue
            # 生成标定结果的可视化图片
            if index_path == 1:
                cv2.imwrite(self.save_path + 'plt_pkl.jpg', img)
            elif index_path == 2:
                cv2.imwrite(self.save_path + 'plt_padding_pkl.jpg', img)

    def padding_pkl(self):
        """
            函数功能：对由相机近段畸变导致标定结果缺失的部分值进行填充
            填充逻辑如下：
                1)去依次增大步幅,寻找左、下、右、上非0值;
                2)满足左右或者上下搜索到值，进行插值平均;
                3)当(没有搜索到成对的非0值时,且已超出图像边界） and 步长超出阈值时，采用最新填充值的左边或上边的非零值;
        """
        
        # # 加载相机掩码图
        mask = cv2.imread('../data/mask/' + self.Dev_name.split('/')[-1][0:-4] + '/Station_' + str(self.station_name) + '/mask' + '_' + self.camera_serial + '.bmp', 0)

        f_read = open(self.save_path + str(self.camera_type) + str(self.camera_serial) + '.pkl', 'rb')
        dict = pickle.load(f_read)
        f_read.close()
        h = 1080
        w = 1920

        pc_xy_lonlat = dict['pc_xy_lonlat']
        new_pkl = np.tile(pc_xy_lonlat, (1, 1))

        # 遍历短焦下半部分像素区域
        for j in range(400, h):
            for i in range(w):
                if mask[j][i] == 255:
                    continue

                index = 1
                pix_x, pix_y = i, j

                if new_pkl[pix_x][pix_y].min() == new_pkl[pix_x][pix_y].max() == 0:
                    flag = np.array([0, 0, 0, 0])  # left,down,right,up 边界判断
                    step = [0, 0, 0, 0]  # left,down,right,up 各方向上的搜索步幅
                    while new_pkl[pix_x][pix_y].min() == 0:

                        print(i, j)
                        # 遍历 left
                        if step[0] == 0 and flag[0] == 0:
                            c = pc_xy_lonlat[pix_x - index][pix_y]
                            if 0 > pix_x - index:
                                flag[0] = 1
                            elif pc_xy_lonlat[pix_x - index][pix_y].min() != 0:
                                print("left：", new_pkl[pix_x][pix_y], pc_xy_lonlat[pix_x - index][pix_y])
                                step[0] = index

                        # 遍历down
                        if step[1] == 0 and flag[1] == 0:
                            if 0 > pix_y - index:
                                flag[1] = 1
                            elif pc_xy_lonlat[pix_x][pix_y - index].min() != 0:
                                print("down:", new_pkl[pix_x][pix_y], pc_xy_lonlat[pix_x][pix_y - index])
                                step[1] = index

                        # 遍历right
                        if step[2] == 0 and flag[2] == 0:
                            if pix_x + index >= 1920:
                                flag[2] = 1
                            elif pc_xy_lonlat[pix_x + index][pix_y].min() != 0:
                                print("right：", new_pkl[pix_x][pix_y], pc_xy_lonlat[pix_x + index][pix_y])
                                step[2] = index

                        # 遍历up
                        if step[3] == 0 and flag[3] == 0:
                            if pix_y + index >= 1080:
                                flag[3] = 1
                            elif pc_xy_lonlat[pix_x][pix_y + index].min() != 0:
                                print("up：", new_pkl[pix_x][pix_y], pc_xy_lonlat[pix_x][pix_y + index])
                                step[3] = index
                        
                        # 像素点左右搜索到相应值，取插值平均，并跳出
                        if step[0] != 0 and step[2] != 0:
                            average = step[0] / (step[0] + step[2])
                            new_pkl[pix_x][pix_y] = [(left + right) * average for left, right in
                                                    zip(pc_xy_lonlat[pix_x - step[0]][pix_y],
                                                        pc_xy_lonlat[pix_x + step[2]][pix_y])]
                            print(" new:", new_pkl[pix_x][pix_y])
                            break
                        # 像素点上下搜索到相应值，取插值平均，并跳出
                        elif step[1] != 0 and step[3] != 0:
                            average = step[1] / (step[1] + step[3])
                            new_pkl[pix_x][pix_y] = [(up + down) * average for down, up in
                                                    zip(pc_xy_lonlat[pix_x][pix_y - step[1]],
                                                        pc_xy_lonlat[pix_x][pix_y + step[3]])]
                            print(" new:", new_pkl[pix_x][pix_y])
                            break
                        # (当没有成对方向的搜索值时，且没有搜索值的方向已到达边界,或索引达到阈值)  将左或者上非空值赋予new_pkl跳出
                        if min([f+s for f, s in zip(flag, step)]) != 0 or index >= 500:
                            if (new_pkl[pix_x - 1][pix_y].min() != new_pkl[pix_x - 1][pix_y].max()) and pix_x > 0:
                                new_pkl[pix_x][pix_y] = new_pkl[pix_x-1][pix_y]
                            elif new_pkl[pix_x][pix_y - 1].min() != new_pkl[pix_x][pix_y - 1].max() and pix_y > 0:
                                new_pkl[pix_x][pix_y] = new_pkl[pix_x][pix_y - 1]
                            break
                        index += 1

                        print("index: ", index)
                    print("new data:",new_pkl[pix_x][pix_y])
                else:
                    new_pkl[i][j] = pc_xy_lonlat[i][j]

        # 保存填充后的结果
        new_pkl_save = {"pc_xy_lonlat": new_pkl}
        padding_pkl_path = self.save_path + 'padding_' + str(self.camera_type) + str(self.camera_serial) + '.pkl'
        f_save = open(padding_pkl_path, 'wb')
        pickle.dump(new_pkl_save, f_save)
        f_save.close()
        self.pkl_to_npy(padding_pkl_path)

    def pkl_to_npy(self, path):
        """
            函数功能: 将pkl文件转换为npy文件
            Input : 文件路径
        """
        f_read = open(path, 'rb')
        dict = pickle.load(f_read)
        f_read.close()
        pc_xy_lonlat = dict['pc_xy_lonlat']
        np.save(path.split('.pkl')[0] + '.npy', pc_xy_lonlat)

    def plot_line(self):
        """
            函数功能：根据生成的pkl文件，查找标定结果的X 和 Y 轴与图像边界的交点坐标
        """
        f_read = open('./result/result' + self.camera_serial + '/1_from.pkl', 'rb')
        dict = pickle.load(f_read)
        f_read.close()
        h = 1080
        w = 1920
        line_x = []
        line_y = []
        pc_xy_lonlat = dict['pc_xy_lonlat']
        for j in range(h):
            for i in range(w):
                x = pc_xy_lonlat[i][j][0]
                y = pc_xy_lonlat[i][j][1]
                if -0.05 <= x <= 0.05 and y != 0.0:
                    line_x.append(i)
                if -0.05 <= y <= 0.05 and x != 0.0:
                    line_y.append(j)
        print(max(line_x), min(line_x))
        print(max(line_y), min(line_y))
