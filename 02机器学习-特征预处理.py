import pandas as pd
from sklearn.preprocessing import  MinMaxScaler,StandardScaler

def minmax_demo():#归一化处理
    #出现异常值（最大值或最小值），对归一化影响很大，鲁棒性较差
    data=pd.read_csv('dating.txt')
    data=data.iloc[:,:3]
    print("data:\n",data)

    transfer=MinMaxScaler()

    data_new=transfer.fit_transform(data)
    print("data_new:\n", data_new)
    return None

def stand_demo():#标准化处理
    data = pd.read_csv('dating.txt')
    data = data.iloc[:, :3]
    print("data:\n", data)

    transfer = StandardScaler()

    data_new = transfer.fit_transform(data)
    print("data_new:\n", data_new)
    return None

if __name__ == '__main__':
    #minmax_demo()
    stand_demo()