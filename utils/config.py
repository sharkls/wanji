import numpy as np
import xml.etree.ElementTree as ET

def read_xml(xml_file, station_name):
    """
        函数功能：读取xml配置文件,并将相关信息返回
        input:
            xml_file: 配置文件路径 (类型 : str)
            station_name: 基站号  (类型 : int)

        output:
            [Longitude, Latitude, AngleNorth, levelOfDetail]:   经度, 纬度, 航向角,  瓦片图的划分等级(类型 : [float, float, float, int])
            [Cam_points_from, Map_points_from] : 路段特征点坐标   (类型 : [array, array])
            [Cam_points_fish, Map_points_fish] : 鱼眼特征点坐标   (类型 : [array, array])
            [Cam_points_to, Map_points_to] :     路口特征点坐标   (类型 : [array, array])
            station_name : 基站号  (类型 : int)
    """
    tree = ET.parse(xml_file)
    root = tree.getroot()
    for child in root:
        if child.tag != 'info_station' + str(station_name):
            continue
        for next in child:
            if next.tag == 'fLidarLongitude':
                Longitude = float(next.text)
            elif next.tag == 'fLidarLatitude':
                Latitude = float(next.text)
            elif next.tag == 'fAngleNorthT':
                AngleNorth = float(next.text)
            elif next.tag == 'flevelOfDetail':
                levelOfDetail = int(next.text)
            elif next.tag == 'Cam_from':
                for i in next:
                    if i.tag == 'fCam_points':
                        points = i.text.split(',')
                        Cam_points_from = np.mat([[int(points[0]), int(points[1])], [int(points[2]), int(points[3])],
                                                [int(points[4]), int(points[5])], [int(points[6]), int(points[7])]])
                    elif i.tag == 'fmap_points':
                        points = i.text.split(',')
                        Map_points_from = np.mat([[int(points[0]), int(points[1])], [int(points[2]), int(points[3])],
                                                [int(points[4]), int(points[5])], [int(points[6]), int(points[7])]])
                    else:
                        print('Failed to parse unknown information')
            elif next.tag == 'Cam_fish':
                for i in next:
                    if i.tag == 'fCam_points':
                        points = i.text.split(',')
                        Cam_points_fish = np.mat([[int(points[0]), int(points[1])], [int(points[2]), int(points[3])],
                                                  [int(points[4]), int(points[5])], [int(points[6]), int(points[7])]])
                    elif i.tag == 'fmap_points':
                        points = i.text.split(',')
                        Map_points_fish = np.mat([[int(points[0]), int(points[1])], [int(points[2]), int(points[3])],
                                                  [int(points[4]), int(points[5])], [int(points[6]), int(points[7])]])
                    else:
                        print('Failed to parse unknown information')
            elif next.tag == 'Cam_to':
                for i in next:
                    if i.tag == 'fCam_points':
                        points = i.text.split(',')
                        Cam_points_to = np.mat([[int(points[0]), int(points[1])], [int(points[2]), int(points[3])],
                                                  [int(points[4]), int(points[5])], [int(points[6]), int(points[7])]])
                    elif i.tag == 'fmap_points':
                        points = i.text.split(',')
                        Map_points_to = np.mat([[int(points[0]), int(points[1])], [int(points[2]), int(points[3])],
                                                  [int(points[4]), int(points[5])], [int(points[6]), int(points[7])]])
                    else:
                        print('Failed to parse unknown information')
            else:
                pass
    return [[Longitude, Latitude, AngleNorth, levelOfDetail], [Cam_points_from, Map_points_from], [Cam_points_fish, Map_points_fish],
            [Cam_points_to, Map_points_to], station_name]

if __name__ == "__main__":
    config = read_xml("../config/Dev126.xml", 0)
    for k, i in enumerate(config):
        if k == 1:
            print(type(i[0]), i[0])
        else:
            print(type(i), i)



