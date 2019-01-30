# 导入必要的包
import cv2

class SimplePreprocessor:
	def __init__(self, width, height, inter=cv2.INTER_AREA):
		# 存储调整大小时使用的目标图像的宽度、高度和插值方法
		self.width = width
		self.height = height
		self.inter = inter
		
	def preprocess(self, image):
		# 将图像调整到固定大小，忽略纵横比
		return cv2.resize(image, (self.width, self.height),interpolation=self.inter)
