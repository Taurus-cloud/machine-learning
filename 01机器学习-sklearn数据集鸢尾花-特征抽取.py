import jieba
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import jieba

def datasets_demo():#载入数据集
    iris=load_iris()
    print("鸢尾花数据集:\n",iris)
    print("查看数据集描述:\n", iris["DESCR"])
    print("查看特征值的名字：\n",iris.feature_names)
    print("查看特征值：\n",iris.data.shape)

    x_train,x_test,y_train,y_test=train_test_split(iris.data,iris.target,test_size=0.2,random_state=22)
    print("训练集的特征值：\n",x_train,x_train.shape)
    return None

def dict_demo():#one-hot编码转化，对文本进行特征抽取
    data=[{'city':'北京','temperature':100},{'city': '上海', 'temperature':60},{'city': '深圳', 'temperature': 30}]
    transfer=DictVectorizer(sparse=False)
    data_new=transfer.fit_transform(data)
    print("data_new:\n",data_new)
    print("特征名字：\n",transfer.get_feature_names_out())
    return None

def count_demo():#文本特征抽取，计算词汇出现次数
    data=["life is short,i like like python","life is too long,i dislike python"]
    transfer=CountVectorizer()
    data_new=transfer.fit_transform(data)
    print("data_new:\n",data_new.toarray())
    print("特征名字：\n", transfer.get_feature_names_out())
    return None

def count_chinese_demo():#中文特征抽取
    data=["我 爱 北京 天安门","天安门 上 太阳 升"]
    transfer=CountVectorizer()
    data_new=transfer.fit_transform(data)
    print("data_new:\n",data_new.toarray())
    print("特征名字：\n", transfer.get_feature_names_out())
    return None

def cut_word(text):#文本分割
    return " ".join(list(jieba.cut(text)))

def count_chinese_demo2():#中文特征抽取2
    data=["一种还是一种今天很残酷，明天更残酷，后天很美好，但绝对大部分是死在明天晚上，所以每个人不要放弃今天。",
          "我们看到的从很远星系来的光是在几百万年之前发出的，这样当我们看到宇宙时，我们是在看它的过去。",
          "如果只用一种方式了解某样事物，你就不会真正了解它。了解事物真正含义的秘密取决于如何将其与我们所了解的事物相联系。"]
    data_new=[]
    for sent in data:
        data_new.append(cut_word(sent))
    transfer = CountVectorizer(stop_words=["一种","所以"])
    data_finally = transfer.fit_transform(data_new)
    print("data_new:\n", data_finally.toarray())
    print("特征名字：\n", transfer.get_feature_names_out())
    return None

def tfidf_demo():#用TF-IDF的方法进行文本特征抽取
    data = ["一种还是一种今天很残酷，明天更残酷，后天很美好，但绝对大部分是死在明天晚上，所以每个人不要放弃今天。",
            "我们看到的从很远星系来的光是在几百万年之前发出的，这样当我们看到宇宙时，我们是在看它的过去。",
            "如果只用一种方式了解某样事物，你就不会真正了解它。了解事物真正含义的秘密取决于如何将其与我们所了解的事物相联系。"]
    data_new = []
    for sent in data:
        data_new.append(cut_word(sent))
    transfer = TfidfVectorizer(stop_words=["一种", "所以"])
    data_finally = transfer.fit_transform(data_new)
    print("data_new:\n", data_finally.toarray())
    print("特征名字：\n", transfer.get_feature_names_out())
    return None

if __name__=="__main__":
    # 代码1 datasets_demo()
    # 代码2 dict_demo()
    # 代码3 count_demo()
    # 代码4 count_chinese_demo()
    # 代码5 count_chinese_demo2()
    # 代码6 tfidf_demo()

