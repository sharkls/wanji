import time
import argparse
import sys
sys.path.append('../')
from utils.Tiles import Tile_Map
from utils.config import read_xml
from concurrent import futures
from utils.mosaic import make_map

def cal_xy(arg):
    config, opt, Id_list, X_list, Y_list, camera_serial = arg
    Map = Tile_Map(config, [Id_list, X_list, Y_list], camera_serial=camera_serial, opt=opt)

    M1 = Map.Tile_icon()  # 生成Transform_later.png
    Map.plt_cam()  # 将瓦片图映射到图像上,生成Make_cam1.png
    Map.image_compose()  # 计算每块瓦片图的坐标并存储
    Map.Save_pkl()  # 保存-+++++


    # 绘制初始pkl文件信息
    if opt.camera_type[camera_serial] == 'fisheye':
        XXX = '0'  # 0为每隔10m绘制不同颜色,2为具体某个像素值对应的xy
    else: 
        XXX = '1'  # 1为将图像中的xy空值输出到图像中
    Map.Plt_pkl(XXX, pointx=739, pointy=803, index_path=1)

    if opt.camera_type[camera_serial] == 'short':
        # 填充pkl中缺失像素的点的信息
        Map.padding_pkl()

        # 绘制填充后pkl文件信息
        XXX = '1'  # XXX，将pkl绘制到图像上，0为每隔10m绘制不同颜色， 1为将图像中的xy空值输出到图像中,2为具体某个像素值对应的xy
        Map.Plt_pkl(XXX, pointx=739, pointy=803, index_path=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--Dir_path', type=str, default='../config/Dev_qingdao_3.xml', help='相机配置文件路径')
    parser.add_argument('--img_path', type=str, default='../data/images/qingdao', help='瓦片图文件路径')
    parser.add_argument('--save_path', type=str, default='../output/', help='保存文件路径')
    parser.add_argument('--camera_nums', type=int, default=3, help='标定相机数量')
    parser.add_argument('--tiles_size', type=int, default=256, help='瓦片图大小')
    parser.add_argument('--camera_type', type=list, default=['short', 'fisheye', 'short'], help='标定相机类型')
    parser.add_argument('--use', type=str, default='one', help='使用两个或一个透视矩阵')
    parser.add_argument('--max_workers', type=int, default=8, help='最大进程数')
    parser.add_argument('--station_name', type=int, default=0, help='拼接哪个基站的瓦片图')
    parser.add_argument('--perceive', type=int, default='140', help='感知范围（上下左右）/米')
    parser.add_argument('--mask', type=bool, default=True, help='是否使用瓦片图的掩码图')
    parser.add_argument('--multiprocess', type=bool, default=False, help='是否使用多进程实现多相机并行标定')
    parser.add_argument('--single_camera', type=bool, default=False, help='是否进行相机的依次标定')

    opt = parser.parse_args()

    # 拼接瓦片图并保留瓦片图对应索引
    config = read_xml(opt.Dir_path, opt.station_name)
    Id_list, X_list, Y_list = make_map(config, opt)

    start = time.time()

    arg = []
    for No_camera in range(opt.camera_nums):
        arg.append([config, opt, Id_list, X_list, Y_list, No_camera])

        # 运行内存不足以支撑多进程时采用循环进行
        if opt.single_camera:
            cal_xy(arg[-1])

    # 单个相机测试, 0: 路段  1:鱼眼 2:路口
    cal_xy(arg[0])

    # 多相机联合标定
    if opt.multiprocess :
        workers = min(opt.max_workers, opt.camera_nums)
        with futures.ProcessPoolExecutor(workers) as executor:     # 采用多进程, 若采用多线程修改为ThreadPoolExecutor
            executor.map(cal_xy, arg)

    end = time.time()
    print("haoshi:", end-start)