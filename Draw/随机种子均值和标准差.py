import numpy as np

# 输入数据
data1 = [66.99, 66.67, 66.85, 66.64, 67.42, 66.32, 66.21, 65.93, 64.88, 64.49]
data2 = [70.01, 70.61, 69.87, 70.54, 69.69, 70.44, 70.43, 70.32, 71.22, 71.11]

# 计算均值
mean1 = np.mean(data1)
mean2 = np.mean(data2)

# 计算标准差
std_dev1 = np.std(data1)
std_dev2 = np.std(data2)

# 输出结果
print(f"均值: {mean1}")
print(f"标准差: {std_dev1}")

print(f"均值: {mean2}")
print(f"标准差: {std_dev2}")