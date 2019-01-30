# 导入必要的包
import numpy as np
import cv2
import os

class SimpleDatasetLoader:
	def __init__(self,preprocessors=None):
		# 存储图像预处理程序
		self.preprocessors = preprocessors
		
		# 如果预处理器为None，则将它们初始化为空列表
		if self.preprocessors is None:
			self.preprocessors = []
			
	def load(self,imagePaths,verbose=-1):
		# 初始化特征和标签列表
		data = []
		labels = []
		
		# 循环输入图像
		for(i,imagePath) in enumerate(imagePaths):
			# 假设路径具有以下格式，加载图像并提取类标签:
			# /path/to/dataset/{class}/{image}.jpg
			image = cv2.imread(imagePath)
			label = imagePath.split(os.path.sep)[-2]
			# 检查我们的预处理器是否为None
			if self.preprocessors is not None:
				# 遍历预处理程序并将每个预处理程序应用于图像
				for p in self.preprocessors:
					image = p.preprocess(image)
			
			# 通过更新数据列表和标签，将我们处理过的图像视为一个“特征向量”
			data.append(image)
			labels.append(label)
			# 显示每个“详细”图像的更新
			if verbose > 0 and i > 0 and (i + 1) % verbose == 0:
				print("[INFO] processed {}/{}".format(i + 1,len(imagePaths)))
		
		# 返回数据和标签的元组
		return (np.array(data), np.array(labels))