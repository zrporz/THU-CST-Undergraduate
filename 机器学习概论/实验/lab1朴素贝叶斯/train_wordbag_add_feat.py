import json
import numpy as np
from sklearn.model_selection import KFold
from tqdm import tqdm
import matplotlib.pyplot as plt
seed = 123
np.random.seed(seed)
model_type = "word_bag"
feat_dim = 100
predict_zeros = False
ratio = 1.0
# alpha_list = [0.1**i for i in range(30)]
alpha = 1e-50
feat_time = False
feat_encoding = False
feat_xmailer = False
feat_recieved= False
weight = 2
def make_word_set(data,from_type="content"):
    if from_type == "date_info":
        day_set = set()
        hour_set = set()
        for _,key in tqdm(enumerate(data)):
            if data[key][from_type]:
                day_set.add(data[key][from_type][0])
                hour_set.add(data[key][from_type][1])
            else:
                day_set.add("None")
                hour_set.add("None")
        return day_set, hour_set
    elif from_type=="content":
        word_set = set()
        for _,key in tqdm(enumerate(data)):
            for content in data[key][from_type]:
                for word in content:
                    word_set.add(word)
        return word_set
    else:
        word_set = set()
        for _,key in tqdm(enumerate(data)):
            if data[key][from_type]:
                word_set.add(data[key][from_type])
            else:
                word_set.add("None")
        return word_set
def split_dataset(data_list):
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    split_dataset = []
    for train_val_index, test_index in kf.split(data_list):
        train_val_data = [data_list[index] for index in train_val_index ] # 获取训练+验证集的数据
        test_data = [data_list[index] for index in test_index ]  # 获取测试集的数据
        
        # 分割训练+验证集为训练集和验证集
        train_len = len(train_val_data)
        train_data, val_data = np.split(np.random.permutation(train_val_data), [int(0.75 * train_len)])

        # 将训练集、验证集和测试集作为字典的值存储到 split_dataset 列表中
        # import pdb; pdb.set_trace()
        split_dataset.append({"train": train_data, "val": val_data, "test": test_data})

    return split_dataset
class BayesClassifier:
    def __init__(self, word_set:set, spam_num:int,ham_num:int,alpha=0,day_set = None,hour_set = None,encoding_set = None,xmailer_set = None,recieved_set = None) -> None:
        self.word_set = word_set

        self.spam_likelihood = {word: [alpha/(spam_num + 2*alpha) for _ in range(feat_dim)] for word in word_set}
        self.ham_likelihood = {word: [alpha/(ham_num + 2*alpha) for _ in range(feat_dim)] for word in word_set}
        aux_feat_key = ["__day__","__hour__","__encode__","__xmail__","__received__"]
        for aux in aux_feat_key:
            self.spam_likelihood[aux] = {}
            self.ham_likelihood[aux] = {}


        self.spam_num = spam_num
        self.ham_num = ham_num
        # self.train_data = train_data
        # self.val_data = val_data
        # self.test_data = test_data
        self.spam_prior = spam_num/(spam_num+ham_num)
        self.ham_prior = ham_num/(spam_num+ham_num)
        self.alpha = alpha
        self.day_set = day_set
        self.hour_set = hour_set
        self.encoding_set = encoding_set
        self.xmailer_set = xmailer_set
        self.recieved_set = recieved_set

    def generate_word_bag(self, data, return_all=False):
        if return_all:
            email_word_bag = {word: 0 for word in self.word_set}
        else:
            email_word_bag = {}
        for content in data["content"]:
            for word in content:
                # if model_type == "word_bag":
                if word not in email_word_bag:
                    email_word_bag[word] = 1
                else:
                    email_word_bag[word] += 1
        return email_word_bag
    def train_likelihood(self, train_data):
        aux_feat_key = ["__day__","__hour__","__encode__","__xmail__","__received__"]
        for _,data in tqdm(train_data):
            email_word_bag = self.generate_word_bag(data)
            date_key = data['date_info'][0] if data['date_info'] else "None"
            hour_key = data['date_info'][1] if data['date_info'] else "None"
            encoding_key = data['encoding'] if data['encoding'] else "None"
            revieved_key = data['recieved_info'] if data['recieved_info'] else "None"
            xmailer_key = data['xmailer_info'] if data['xmailer_info'] else "None"
            if data['label'] == "ham":
                for word, appear in email_word_bag.items():
                    appear = min(appear,feat_dim-1)
                    self.ham_likelihood[word][appear] += 1/(self.ham_num + 2*self.alpha )
                if date_key not in self.ham_likelihood["__day__"]:
                    self.ham_likelihood["__day__"][date_key] = 0
                if hour_key not in self.ham_likelihood["__hour__"]:
                    self.ham_likelihood["__hour__"][hour_key] = 0
                if encoding_key not in self.ham_likelihood["__encode__"]:
                    self.ham_likelihood["__encode__"][encoding_key] = 0
                if revieved_key not in self.ham_likelihood["__received__"]:
                    self.ham_likelihood["__received__"][revieved_key] = 0
                if xmailer_key not in self.ham_likelihood["__xmail__"]:
                    self.ham_likelihood["__xmail__"][xmailer_key] = 0
                self.ham_likelihood["__day__"][date_key] += 1/(self.ham_num + 2*self.alpha )
                self.ham_likelihood["__hour__"][hour_key] += 1/(self.ham_num + 2*self.alpha )
                self.ham_likelihood["__encode__"][encoding_key] += 1/(self.ham_num + 2*self.alpha )
                self.ham_likelihood["__received__"][revieved_key] += 1/(self.ham_num + 2*self.alpha )
                self.ham_likelihood["__xmail__"][xmailer_key] += 1/(self.ham_num + 2*self.alpha )
            elif data['label'] == "spam":
                for word, appear in email_word_bag.items():
                    appear = min(appear,feat_dim-1)
                    self.spam_likelihood[word][appear] += 1/(self.spam_num + 2*self.alpha )
                if date_key not in self.spam_likelihood["__day__"]:
                    self.spam_likelihood["__day__"][date_key] = 0
                if hour_key not in self.spam_likelihood["__hour__"]:
                    self.spam_likelihood["__hour__"][hour_key] = 0
                if encoding_key not in self.spam_likelihood["__encode__"]:
                    self.spam_likelihood["__encode__"][encoding_key] = 0
                if revieved_key not in self.spam_likelihood["__received__"]:
                    self.spam_likelihood["__received__"][revieved_key] = 0
                if xmailer_key not in self.spam_likelihood["__xmail__"]:
                    self.spam_likelihood["__xmail__"][xmailer_key] = 0
                self.spam_likelihood["__day__"][date_key] += 1/(self.spam_num + 2*self.alpha )
                self.spam_likelihood["__hour__"][hour_key] += 1/(self.spam_num + 2*self.alpha )
                self.spam_likelihood["__encode__"][encoding_key] += 1/(self.spam_num + 2*self.alpha )
                self.spam_likelihood["__received__"][revieved_key] += 1/(self.spam_num + 2*self.alpha )
                self.spam_likelihood["__xmail__"][xmailer_key] += 1/(self.spam_num + 2*self.alpha )

        for word in self.word_set:
            self.ham_likelihood[word][0] = 1-sum(self.ham_likelihood[word][1:])
            self.spam_likelihood[word][0] = 1-sum(self.spam_likelihood[word][1:])
            
        print("training done")
                # for word in email_word_set:
                #     self.spam_likelihood[word] += 1/self.spam_num

    def predict_single_data(self,intput_data):
        # email_word_set = set()
        ham_joint_likelihood = 0
        spam_joint_likelihood = 0
        # email_word_bag = {word: 0 for word in self.word_set}
        # for content in intput_data["content"]:
        #     for word in content:
        #         email_word_set.add(word)
        email_word_bag = self.generate_word_bag(intput_data,return_all = predict_zeros)
        for word,appear in email_word_bag.items():
            appear = min(appear,feat_dim-1)
            ham_joint_likelihood += np.log(self.ham_likelihood[word][appear])
            spam_joint_likelihood += np.log(self.spam_likelihood[word][appear])
        # print(data)
        date_key = intput_data['date_info'][0] if intput_data['date_info'] else "None"
        hour_key = intput_data['date_info'][1] if intput_data['date_info'] else "None"
        encoding_key = intput_data['encoding'] if intput_data['encoding'] else "None"
        revieved_key = intput_data['recieved_info'] if intput_data['recieved_info'] else "None"
        xmailer_key = intput_data['xmailer_info'] if intput_data['xmailer_info'] else "None"
        if feat_time and date_key in self.ham_likelihood["__day__"] and date_key in self.spam_likelihood["__day__"] and hour_key in self.ham_likelihood["__hour__"] and hour_key in self.spam_likelihood["__hour__"]:
            ham_joint_likelihood += np.log(self.ham_likelihood["__day__"][date_key])*weight
            ham_joint_likelihood += np.log(self.ham_likelihood["__hour__"][hour_key])*weight
            spam_joint_likelihood += np.log(self.spam_likelihood["__day__"][date_key])*weight
            spam_joint_likelihood += np.log(self.spam_likelihood["__hour__"][hour_key])*weight
        if feat_encoding and encoding_key in self.ham_likelihood["__encode__"] and encoding_key in self.spam_likelihood["__encode__"]:
            ham_joint_likelihood += np.log(self.ham_likelihood["__encode__"][encoding_key])*weight
            spam_joint_likelihood += np.log(self.spam_likelihood["__encode__"][encoding_key])*weight
        if feat_recieved and  revieved_key in self.ham_likelihood["__received__"] and revieved_key in self.spam_likelihood["__received__"]:
            ham_joint_likelihood += np.log(self.ham_likelihood["__received__"][revieved_key])*weight
            spam_joint_likelihood += np.log(self.spam_likelihood["__received__"][revieved_key])*weight
        if feat_xmailer and xmailer_key in self.ham_likelihood["__xmail__"] and xmailer_key in self.spam_likelihood["__xmail__"]:
            # print(self.ham_likelihood["__xmail__"].keys())
            ham_joint_likelihood += np.log(self.ham_likelihood["__xmail__"][xmailer_key])*weight
            spam_joint_likelihood += np.log(self.spam_likelihood["__xmail__"][xmailer_key])*weight


        if spam_joint_likelihood + np.log(self.spam_prior) > ham_joint_likelihood + np.log(self.ham_prior):
            answer = "spam"
        else:
            answer = "ham"
        return answer, max(spam_joint_likelihood + np.log(self.spam_prior),ham_joint_likelihood + np.log(self.ham_prior))
    
    def test(self,test_data,test_id):
        test_num = len(test_data)
        confusions_matrix = {"spam":{"spam":0,"ham":0},"ham":{"spam":0,"ham":0}}
        score_sort = []
        print(f"Test {test_id}")
        print(f"==============")
        for _,data in tqdm(test_data):
            answer, prob = self.predict_single_data(data)
            confusions_matrix[data['label']][answer] += 1
            score_sort.append((data['label'],prob,answer))
        score_sort.sort(key=lambda x:x[1],reverse=True)
        num = 0
        for score in score_sort:
            if score[1]==-float("inf"):
                num+=1
        print(score_sort[-1])
        print(f"num:{num},{len(score_sort)},ratio:{num/len(score_sort)}")
        s0 = 0
        n0 = 0
        for id in range(len(score_sort)):
            if score_sort[id][0] == "ham":
                s0 += id+1
                n0 += 1
        auc = (s0 - n0*(n0+1)/2)/(n0*(test_num-n0))
        # 绘制 aoc 曲线图
        x = [i/n0 for i in range(1,n0+1)]
        y = []
        rate = 0
        for i in range(len(score_sort)):
            if score_sort[i][0] == "spam":
                rate += 1/(test_num-n0)
            else:
                y.append(rate)
        plt.plot(x,y, label = f"Test {test_id}")
        plt.fill_between(x, y, 0, alpha=0.1)  # 在曲线和x轴之间添加阴影
        plt.ylabel("True Positive Rate")
        plt.xlabel("False Positive Rate")
        plt.title("ROC Curve")
        plt.savefig("auc.png")
        # print(n0, test_num-n0)
        return {"accuracy":(confusions_matrix["spam"]["spam"]+confusions_matrix["ham"]["ham"])/test_num,
                "precision":confusions_matrix["spam"]["spam"]/(confusions_matrix["spam"]["spam"]+confusions_matrix["ham"]["spam"]),
                "recall":confusions_matrix["spam"]["spam"]/(confusions_matrix["spam"]["spam"]+confusions_matrix["spam"]["ham"]),
                "F1": 2*confusions_matrix["spam"]["spam"]/(2*confusions_matrix["spam"]["spam"]+confusions_matrix["spam"]["ham"]+confusions_matrix["ham"]["spam"]),
                "AUC":auc}
        # correct_num = 0
        # for _,data in tqdm(test_data):
        #     answer = self.predict_single_data(data)
        #     if answer == data['label']:
        #         correct_num += 1
        # return correct_num/len(test_data)
        
if __name__ == "__main__":
    # for alpha in alpha_list:
    with open("output.json","r",encoding='utf-8') as f:
        data = json.load(f)
        word_set = make_word_set(data)
        day_set,hour_set = make_word_set(data,"date_info")
        encoding_set = make_word_set(data,"encoding")
        xmailer_set = make_word_set(data,"xmailer_info")
        recieved_set = make_word_set(data,"recieved_info")
        data_list = list(data.items())
        split_data_list = split_dataset(data_list[:int(len(data_list)*ratio)])
        ham_num = 0
        spam_num = 0
        for _,key in tqdm(enumerate(data)):
            if data[key]["label"] == "spam":
                spam_num += 1
            elif data[key]["label"] == "ham":
                ham_num += 1
            else:
                raise ValueError(data[key]["label"])
    id = 0
    result_list = {"accuracy":[],
                "precision":[],
                "recall":[],
                "F1": [],
                "AUC":[]}
    for split_data_ in split_data_list:
        # import pdb; pdb.set_trace()
        classifier = BayesClassifier(word_set,spam_num,ham_num,alpha,day_set,hour_set,encoding_set,xmailer_set,recieved_set)
        classifier.train_likelihood(np.concatenate((split_data_['train'],split_data_['val']),axis=0))
        val_results =  classifier.test(split_data_['test'],id)
        
        # classifier.train_likelihood(np.concatenate((split_data_['train'],split_data_['val']),axis=0))
        results = classifier.test(split_data_['test'],id)
        for key,result in results.items():
            result_list[key].append(result)
        id+=1
        print(results)
    for key,result in result_list.items():
        print(f"mean {key}: {sum(result)/len(result)}")
    with open("result.json","w") as f:
        json.dump(result_list,f,indent=4)
        # import pdb; pdb.set_trace()