# 导入必要的包

# k-NN算法的实现
from sklearn.neighbors import KNeighborsClassifier 
# 用于将字符串表示的标签转换为整数，其中每个类标签有一个惟一的整数
from sklearn.preprocessing import LabelEncoder 
# 用于创建训练和测试分割
from sklearn.model_selection import train_test_split
# 用于评估分类器的性能，并将格式化良好的结果表打印到控制台
from sklearn.metrics import classification_report
from pyimagesearch.preprocessing.SimplePreprocessor import SimplePreprocessor
from pyimagesearch.datasets.SimpleDatasetLoader import SimpleDatasetLoader
from imutils.paths import paths
import argparse

# 构造参数parse并解析参数
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,help="path to input dataset")
ap.add_argument("-k", "--neighbors", type=int, default=1,help="# of nearest neighbors for classification")
ap.add_argument("-j", "--jobs", type=int, default=-1,help="# of jobs for k-NN distance (-1 uses all available cores)")
args = vars(ap.parse_args())
# 抓取我们将要描述的图像列表
print("[INFO] loading images...")
imagePaths = list(paths.list_images(args["dataset"]))

# 初始化图像预处理程序，从磁盘加载数据集，并重新塑造数据矩阵
sp = SimplePreprocessor(32, 32)
sdl = SimpleDatasetLoader(preprocessors=[sp])
(data, labels) = sdl.load(imagePaths, verbose=500)
data = data.reshape((data.shape[0], 3072))

# 显示一些关于图像内存消耗的信息
print("[INFO] features matrix: {:.1f}MB".format(data.nbytes / (1024 * 1000.0)))

# 将标签编码为整数
le = LabelEncoder()
labels = le.fit_transform(labels)
 
 # 使用75%的数据用于训练，剩下的25%用于测试，将数据划分为训练和测试部分
(trainX, testX, trainY, testY) = train_test_split(data, labels,test_size=0.25, random_state=42)

# 训练和评估一个k-NN分类器的原始像素强度
print("[INFO] evaluating k-NN classifier...")
model = KNeighborsClassifier(n_neighbors=args["neighbors"],n_jobs=args["jobs"])
model.fit(trainX, trainY)
print(classification_report(testY, model.predict(testX),target_names=le.classes_))



