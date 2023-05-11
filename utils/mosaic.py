import glob
import os
import sys
sys.path.append('../')
import PIL.Image as Image
import cv2
import argparse
from utils.tiles2lonlat import TilesSystem
from utils.config import read_xml


def make_map(config, opt):
    """
        拼接瓦片图，保存瓦片图ID和XY最小坐标值
    """

    Id_list, X_list, Y_list = [], [], [] # 瓦片图ID存放列表、 X最值存放列表 、 Y值存放列表

    test = TilesSystem()     # 实例化坐标转换公式
    Longitude = config[0][0] # 经度
    Latitude = config[0][1]  # 维度
    AngleNorth = config[0][2] # 航向角
    levelOfDetail = config[0][3] # 瓦片图等级
    IMAGE_SIZE = opt.tiles_size # 瓦片图大小
    save_path = opt.save_path   # 瓦片图存放路径

    # 将基站经纬度和航向角转换为绝对像素坐标
    pixelX, pixelY = test.LatLongToPixelXY(Longitude, Latitude, levelOfDetail)
    # 根据绝对像素坐标确定瓦片图序号
    stationX, stationY = test.PixelXYToTileXY(pixelX, pixelY)
    imgList = os.listdir(opt.img_path)  # 瓦片图的路径
    imgList.sort()

    # 确定存储路径
    save_path = save_path + 'result' + str(len(glob.glob(save_path + 'result*'))) + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # 读取瓦片图掩码图
    if opt.mask:
        mask_tile = cv2.imread('../data/mask/' + opt.Dir_path.split('/')[-1][0:-4] + '/mask_tile.png', 0)

    # 遍历瓦片图
    for imgFile in imgList:
        # 获取瓦片图的序号
        file_str = os.path.splitext(imgFile)[0]
        sperate = file_str.split('y')
        x0 = sperate[0].split('x')[1]
        y0 = sperate[1].split('_')[0]

        # 根据瓦片图的序号求对应的绝对像素坐标
        pixelX, pixelY = test.TileXYToPixelXY(float(x0), float(y0))
        # 根据绝对像素坐标求对应的经纬度坐标
        Lon, Lat = test.PixelXYToLatLong(pixelX, pixelY, 23)
        # 根据经纬度坐标求xyz坐标
        xyz = test.lon_lat_toxy(Lon, Lat, Longitude, Latitude, AngleNorth)

        # 利用X坐标进行瓦片图的筛除,对要保留的区域保存其瓦片图的序号
        if config and imgFile:
            # 超出区域的瓦片图，直接跳过
            if xyz[0][0] > opt.perceive or xyz[0][0] < -opt.perceive:
                continue
            if xyz[0][1] > opt.perceive or xyz[0][1] < -opt.perceive:
                continue
            
            Id_list.append([int(x0), int(y0), xyz[0][0], xyz[0][1], Lon, Lat])
            X_list.append(int(x0))
            Y_list.append(int(y0))

        # else:
        #     if xyz[0][0] > 5 or xyz[0][0] < -100:
        #         continue
        #     Id_list.append([int(x0), int(y0), xyz[0][0], xyz[0][1], Lon, Lat])
        #     X_list.append(int(x0))
        #     Y_list.append(int(y0))

    tilesX_max, tilesX_min, tilesY_max, tilesY_min = max(X_list), min(X_list), max(Y_list), min(Y_list)

    # 保存瓦片图所保留的区域序号的最小最大值
    del X_list[:], Y_list[:]
    X_list, Y_list = [tilesX_min, tilesX_max], [tilesY_min, tilesY_max]

    COLUMN = int(str(tilesX_max - 1)[-4:]) - int(str(tilesX_min - 1)[-4:]) + 10
    IMAGE_ROW = int(str(tilesY_max - 1)[-4:]) - int(str(tilesY_min - 1)[-4:]) + 10
    to_image = Image.new('RGB', (COLUMN * IMAGE_SIZE, IMAGE_ROW * IMAGE_SIZE))

    # 遍历要保留区域的瓦片图，并进行拼接
    useless_Id = []   # 存放不在指定区域的瓦片图Id号
    for index, Id in enumerate(Id_list):
        # print(index)
        imgFile = 'x' + str(Id[0]) + 'y' + str(Id[1]) + '.png'
        tempDir = os.path.join(opt.img_path, imgFile)
        if os.path.isdir(tempDir):   # os.path.isdir需要绝对路径
            continue
        x_id = int(str(Id[0])[-4:]) - int(str(tilesX_min - 1)[-4:])
        y_id = int(str(Id[1])[-4:]) - int(str(tilesY_min - 1)[-4:])

        # 若瓦片角点像素在掩码图上的空白区域，直接跳过
        if opt.mask:
            x_min, y_min = x_id * IMAGE_SIZE, y_id * IMAGE_SIZE
            x_max, y_max = x_min + 256, y_min + 256
            list = [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]]
            flag = 0
            for angular_point in list:
                if mask_tile[angular_point[1], angular_point[0]] == 255:
                    flag += 1
            if flag >= 4:
                useless_Id.append(Id)
                continue
        print("x_id:", x_id, "y_id:", y_id)

        from_image = Image.open(os.path.join(tempDir))
        to_image.paste(from_image, (x_id * IMAGE_SIZE, y_id * IMAGE_SIZE))

    Id_list = [id for id in Id_list if id not in useless_Id]
    # 将要保留区域的瓦片图拼接后的图像保存
    tiles_img_path = save_path + 'Make_tile1.png'
    to_image.save(tiles_img_path)
    plt_map(Id_list, X_list, Y_list, save_path, stationX, stationY)
    return Id_list, X_list, Y_list

# 将瓦片图的x和y坐标贴在图像上
def plt_map(Id_list, X_list, Y_list, path, stationX, stationY):
    img = cv2.imread(path + 'Make_tile1.png')

    # 遍历要保留区域的瓦片图，并绘制瓦片图对应的X和Y坐标信息
    for Id in Id_list:
        x_id = int(Id[0] - (X_list[0] - 1))
        y_id = int(Id[1] - (Y_list[0] - 1))
        xmin, ymin = x_id * 256, y_id * 256
        xmax, ymax = xmin + 256, ymin + 256
        pt1 = (xmin, ymin)
        pt2 = (xmax, ymax)
        # 根据瓦片图左上右下坐标画瓦片图矩形框(red)
        cv2.rectangle(img, pt1, pt2, (0, 255, 0), 3)
        x_ = str(round(Id[2], 3))
        y_ = str(round(Id[3], 3))
        lon_ = str(round(Id[4], 5)).split('.')[1]
        lat_ = str(round(Id[5], 5)).split('.')[1]
        font = cv2.FONT_HERSHEY_SIMPLEX

        # 瓦片图序号等基站序号用蓝色, x坐标位于基站区域用红色,其他区域为绿色
        if int(Id[0]) == int(stationX) and int(Id[1]) == int(stationY):
            cv2.putText(img, '{} {}'.format(x_, y_), (int((xmax - xmin) / 4 + xmin), int((ymax - ymin) / 3 + ymin)),
                        font, 0.7, (0, 0, 255), 4)
            cv2.putText(img, '{} {}'.format(lon_, lat_),
                        (int((xmax - xmin) / 4 + xmin), int((ymax - ymin) * 2 / 3 + ymin)), font, 0.7, (0, 0, 255), 4)
        elif -5 <= Id[2] <= 5:
            cv2.putText(img, '{} {}'.format(x_, y_),
                        (int((xmax - xmin) / 4 + xmin), int((ymax - ymin) / 3 + ymin)), font, 0.7, (255, 0, 0), 4)
            cv2.putText(img, '{} {}'.format(lon_, lat_),
                        (int((xmax - xmin) / 4 + xmin), int((ymax - ymin) * 2 / 3 + ymin)), font, 0.7, (255, 0, 0), 4)
        else:
            cv2.putText(img, '{} {}'.format(x_, y_),
                        (int((xmax - xmin) / 4 + xmin), int((ymax - ymin) / 3 + ymin)), font, 0.7, (0, 255, 0), 4)
            cv2.putText(img, '{} {}'.format(lon_, lat_),
                        (int((xmax - xmin) / 4 + xmin), int((ymax - ymin) * 2 / 3 + ymin)), font, 0.7, (0, 255, 0), 4)
    # 将填充了坐标的信息的拼接成的瓦片图输出
    cv2.imwrite(path + 'Make_tile2.png', img)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--Dir_path', type=str, default='../config/Dev_qingdao_6.xml', help='相机配置文件路径')
    parser.add_argument('--img_path', type=str, default='../data/images/qingdao', help='瓦片图文件路径')
    parser.add_argument('--save_path', type=str, default='../output/', help='保存文件路径')
    parser.add_argument('--camera_nums', type=int, default=3, help='标定相机数量')
    parser.add_argument('--tiles_size', type=int, default=256, help='瓦片图大小')
    parser.add_argument('--camera_type', type=list, default=['short', 'fisheye', 'short'], help='标定相机类型')
    parser.add_argument('--use', type=str, default='one', help='使用两个或一个透视矩阵')
    parser.add_argument('--perceive', type=int, default='140', help='感知范围（上下左右）')
    parser.add_argument('--mask', type=bool, default=False, help='是否使用瓦片图的掩码图')
    parser.add_argument('--station_name', type=int, default=0, help='拼接那个基站的瓦片图')
    opt = parser.parse_args()

    # 拼接瓦片图并保留瓦片图对应索引
    config = read_xml(opt.Dir_path, opt.station_name)
    Id_list, X_list, Y_list = make_map(config, opt)