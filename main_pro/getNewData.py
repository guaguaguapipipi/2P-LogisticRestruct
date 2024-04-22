import pandas as pd

originalFile = '所有国家的新确诊数据.csv'  # 这个是转置钱的文件需要存放的位置
TFile = '转置的数据.csv'  # 这个是转置之后的文件需要存放的位置

file=open(TFile, 'w', newline='')
df = pd.read_csv(originalFile, header= None, low_memory=False)
# df.values
data = df.values
data = list(map(list,zip(*data)))
data = pd.DataFrame(data)
data.to_csv(file,header=0,index=0)
print("finish")
