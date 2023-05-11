import numpy as np
import math

class TilesSystem(object):
    """
    瓦片图与绝对经纬度之间的转换
    """
    def __init__(self):
        # 常量
        # 赤道半径
        self.equatorialRadius = 6378137
        # 极径
        self.polarRadius = 6356752.314
        self.minLatitude = -85.05112878
        self.maxLatitude = 85.05112878
        self.minLongitude = -180
        self.maxLongitude = 180
        self.R_a = 6378137.00
        self.R_b = 6356752.3142
    def MapSize(self, levelOfDetail):
        """
        确定地图分辨率(m/pixel)
        """
        return 256 << levelOfDetail
    def Clip(self, n, minValue, maxValue):
        """
        选择合适的数据进行计算，避免经纬度or像素坐标越界
        """
        return min(max(n, minValue), maxValue)
    def LatLongToPixelXY(self, longitude, latitude, levelOfDetail):
        """
        将经纬度转换为瓦片地图像素坐标
        """
        mapSize = self.MapSize(levelOfDetail)
        sinLat = math.sin(latitude * math.pi / 180)
        x = (longitude + 180) / 360
        y = 0.5 - math.log((1 + sinLat) / (1 - sinLat)) / (4 * math.pi)
        pixelX = int(x * mapSize + 0.5)
        pixelY = int(y * mapSize + 0.5)
        # print("pixelX = %.4f, pixelY = %.4f"%(pixelX, pixelY))
        return pixelX, pixelY
    def PixelXYToLatLong(self, pixelX, pixelY, levelOfDetail):
        """
        转换为经纬度(绝对的像素位置)
        """
        mapSize = self.MapSize(levelOfDetail)
        x = pixelX / mapSize - 0.5
        y = 0.5 - pixelY / mapSize
        latitude = 90 - 360 * math.atan(math.exp(-y * 2 * math.pi)) / math.pi
        longitude = 360 * x
        return longitude, latitude
    def RefPixelXYToAbsPixelXY(self, ref_pixelX, ref_pixelY, tilesOriginX, tilesOriginY):
        """
        将拼好的瓦片地图子图中的相对像素坐标转为世界瓦片图的绝对像素坐标
        input:在子图中的像素坐标，子图的tiles起点x，y
        output：在完整瓦片图中的绝对像素坐标
        """
        x0 = math.floor(ref_pixelX / 256)
        y0 = math.floor(ref_pixelY / 256)
        abs_tilesX = x0 + tilesOriginX
        abs_tilesY = y0 + tilesOriginY
        abs_pixelX = abs_tilesX * 256 + ref_pixelX - 256 * x0 + 1
        abs_pixelY = abs_tilesY * 256 + ref_pixelY - 256 * y0 + 1
        return abs_pixelX, abs_pixelY
    def abscoordinate(self, abs_pixelX, abs_pixelY):
        """
        docstring
        """
        utm_x = abs_pixelX * 0.0187
        utm_y = abs_pixelY * 0.0187
        return utm_x, utm_y
    def PixelXYToTileXY(self, pixelX, pixelY):
        """
        绝对像素坐标确定瓦片地图序号
        """
        tilesX = pixelX / 256
        tilesY = pixelY / 256
        # print("tilesX = %d, tilesY = %d"%(tilesX, tilesY))
        return tilesX, tilesY
    def TileXYToPixelXY(self, tilesX, tilesY):
        """
        瓦片地图序号确定绝对像素坐标,加256求的是碎片瓦片图右下角点坐标的经纬度，不加256求的是左上角的经纬度
        """
        pixelX = tilesX * 256 #+256
        pixelY = tilesY * 256 #+256
        return pixelX, pixelY

    def lon_lat_toxy(self, lon, lat, longitude, latitude, AngleNorth):
        """
        计算瓦片图在单基站下的位移差
        """
        x = (lon - longitude) * math.pi / 180 * self.R_a * math.cos(latitude * math.pi / 180)
        y = (lat - latitude) * math.pi / 180 * self.R_b
        xyz = np.array([[x, y, -4.2]])  # 旋转
        R_bmp = self.cal_trans(0, 0, AngleNorth * np.pi / 180)
        A = np.dot(R_bmp, xyz[:, :3].T)
        xyz[:, :3] = A.T
        return xyz

    def cal_trans(self, x, y, z):
        R_x = np.array([[1.0, 0.0, 0.0], [0.0, math.cos(x), -1 * math.sin(x)], [0.0, math.sin(x), math.cos(x)]])
        R_y = np.array([[math.cos(y), 0.0, math.sin(y)], [0.0, 1.0, 0.0], [-1 * math.sin(y), 0.0, math.cos(y)]])
        R_z = np.array([[math.cos(z), -1 * math.sin(z), 0.0], [math.sin(z), math.cos(z), 0.0], [0.0, 0.0, 1.0]])
        rotate = np.dot(R_z, R_y)
        rotate = np.dot(rotate, R_x)
        return rotate

    def reCalculateBBS(self, BBS, M):
        M = np.transpose(M)
        for i in range(len(BBS)):
            k1 = BBS[i]
            x = (k1[0] * M[0][0] + k1[1] * M[1][0] + M[2][0]) / (k1[0] * M[0][2] + k1[1] * M[1][2] + M[2][2])
            y = (k1[0] * M[0][1] + k1[1] * M[1][1] + M[2][1]) / (k1[0] * M[0][2] + k1[1] * M[1][2] + M[2][2])
            BBS[i] = (int(x), int(y))
        return BBS