import numpy as np
import glob
import os  
import re
import random
import matplotlib.pyplot as plt
from scipy.stats import logistic
from collections import Counter
import sys
nums = re.compile(r'(\d+)')
def numericalSort(val):
    parts = nums.split(val)
    parts[1::2] = map(int, parts[1::2])
    return parts

def words(path):
    
    all_words = []
    d=[]
    
    for files in sorted(os.listdir(path),key=numericalSort):
        
        if files != 'index.csv':
            d.append(files)
            curr_file = open(path+files,errors='ignore').read().lower() 
#           print(curr_file) 
            data = np.array(re.findall(re.compile(r'[\w]+'),curr_file))
            all_words.append(data)
           
    return (all_words,d)

data_dir = '20newsgroups'
K=int(20) 

(corpus,D) = words('pp4data/' +data_dir+'/')

vocabulary=np.unique(np.concatenate(corpus))

# GIVING numerical values to words
word_dictionary={vocabulary[i]:i for i in range(0,len(vocabulary))}


'''Bag of words 
'''
bag_of_words=np.zeros((len(corpus),len(vocabulary)))
count=0
for docs in corpus:
    c=Counter(docs)
    bag_of_words
    for key,value in c.items():
        bag_of_words[count][word_dictionary[key]]=value
    count+=1
#bag_of_words
np.random.seed(1)

d_n=[] 

i=0
w_n = [word_dictionary[words] for docs in corpus for words in docs]
z_n = [np.random.randint(0,K) for docs in corpus for words in docs]
for docs in corpus:
    for words in docs:
        d_n.append(i)
    i=i+1
phi_n=np.random.permutation(len(w_n))


len(phi_n)

#C_d is counts per document D*K where D is no of documents and K is number of topics
n_topics=len(set(z_n)) #K
n_documents=len(set(d_n)) #D
C_d=np.zeros((n_documents,n_topics))
# C_t is word count per document K*V where K is no of topics and V is vocabulary
n_vocab=len(vocabulary) # V
C_t=np.zeros((n_topics,n_vocab))
P=np.zeros(n_topics)

'''
Creating C_d and C_t from w(n) d(n) and z(n)
'''
# Creating C_d 
for index in range(0,len(w_n)):
    C_d[d_n[index],z_n[index]]+=1
    C_t[z_n[index],w_n[index]]+=1 

def normalize(P):
    return P/sum(P)



''' THE LDA ALGORITHM'''
K=n_topics;alpha=5/K;beta=.01;V=n_vocab
topic_list=range(0,n_topics)
for i in range(0,500):
    for n in range(0,len(w_n)):
        word=w_n[phi_n[n]]
        topic=z_n[phi_n[n]]
        doc=d_n[phi_n[n]]
        C_d[doc][topic]=C_d[doc][topic]-1
        C_t[topic][word]=C_t[topic][word]-1
        k=list(range(0,n_topics))
        sum_C_d=C_d.sum(axis=1)
        sum_C_t=C_t.sum(axis=1)
        C_t_ratio=(C_t[:,word]+beta)/(V*beta+sum_C_t)
        C_d_ratio=(C_d[doc][k]+alpha)/(K*alpha+sum_C_d[doc])
        P=np.array(C_t_ratio*C_d_ratio)
        P=normalize(C_t_ratio*C_d_ratio)
        topic=int(np.random.choice(topic_list,1,p=P))
        z_n[phi_n[n]]=topic
        C_d[doc][topic]=C_d[doc][topic]+1
        C_t[topic][word]=C_t[topic][word]+1


def get_key(val): 
    for key, value in word_dictionary.items(): 
         if val == value:
                return key 
    return "Null"

## Output of the most frequent words
ls1=[["Topic","Word1","Word2","word3","Word4","Word5"]]
rows,cols=C_t.shape
for row in range(0,rows):
    row_list=[row+1]
    value=C_t[row].argsort()[-5:][::-1]
    keys=[get_key(index) for index in value]
    row_list.extend(keys)
    ls1.append(row_list)
np.savetxt("topic_wordss.csv", ls1, delimiter=",",fmt="%s")    


K=n_topics
alpha=5/K
beta=.01
V=n_vocab
for i in range(len(C_d)):
    C_d[i]=(C_d[i]+alpha)/(K*alpha+sum_C_d[i])

#file_label_path =os.path.join(os.getcwd(), 'C://Users//rushabh shah//OneDrive//Desktop//Assignments//ML//programming assignment 4//pp4data//20newsgroups/', "index.csv")

labels=np.genfromtxt('pp4data/'+data_dir+'/index.csv',delimiter=',')                
#labels=np.genfromtxt(label_file,delimiter=',')
#labels.shape

labels=np.delete(labels,0,1)

################# 
'''LOGISTIC FROM PREVIOUS ASSIGNMENT'''
def logistic_regression(X):
    X=np.append(X,labels,axis=1)
    (rows,columns)=X.shape
                           ## X here is the combined data i.e examples and labels in one matrix, 
                                                        ## passing from main function
    mean_accuracy=[[],[],[],[],[],[],[],[],[],[]]
    for p in range(30):

        np.random.shuffle(X)                ## randomnly shuffle data
        split = int(len(X)/3)            ## 1/3 split  

                ## This here is our data seperated from labels
        x = X[:rows,:(columns-1)]    


        label = np.transpose(X)[-1]
                                       ## similarly for labels
        test_data = x[:split]                       ## This here is my testing data i.e acc to the split size 1/3rd here
        test_target = label[:split]
        data = x[split:]                        ### remaining data is used for training
        target = label[split:] 
        alpha =0.01
        for i in range(1,11):
            subsample=int((i)/(10) * len(data))             ## subsamples of increasing size

            t_hat=[]
            train = data[:subsample]
                   ## train is phi(X)
            t = target[:subsample]

            #t = np.concatenate(t)
            train_transpose = np.transpose(train)    
            W =np.zeros(len(train_transpose))       ## wieght vector

            w=np.zeros(len(train_transpose))


            s=0
            while(s<100 ):
                a = np.dot(train,W)
                y = logistic.cdf(a)
                d = t-y
                r = np.multiply(y,1-y)
                w=W                                                  ## calculating gradient and hessian               
                h = -(np.dot(np.dot(train_transpose,np.diag(r)),train) + alpha*np.identity(len(W)))
                H = np.linalg.inv(h)
                gradient = np.dot(train_transpose,d) - alpha*w
                W = w - np.dot(H,gradient)
                s+=1
                if np.linalg.norm(W-w)/np.linalg.norm(w) < .001:
                    break
                t_hat = logistic.cdf(np.dot(test_data,W)) >= 0.5 ## t hat and error for logistic
                mean_accuracy[i-1].append(np.mean(np.equal(test_target,t_hat)))
    return mean_accuracy



mean_accu_lda=logistic_regression(C_d)
mean_accu_bag_of_words=logistic_regression(bag_of_words)



mean_lda=[];mean_bag=[];std_lda=[];std_bag=[]
for k in range(len(mean_accu_bag_of_words)):
    mean_lda.append(np.mean(mean_accu_lda[k]))  
    std_lda.append(np.std(mean_accu_lda[k]))
    mean_bag.append(np.mean(mean_accu_bag_of_words[k]))
    std_bag.append(np.std(mean_accu_bag_of_words[k]))
x_axis = ['0.1','0.2','0.3','0.4','0.5','0.6','0.7','0.8','0.9','1.0']

plt.ylabel("Mean Accuracy")
plt.xlabel("Subsamples")
#plt.errorbar(x_axis,y,yerr = std,color='r')
plt.errorbar(x_axis,mean_lda,yerr = std_lda,label='Bag of words')
plt.errorbar(x_axis,mean_bag,yerr=std_bag,label="LDA")
plt.legend()
plt.savefig('pp4.jpg')
#plt.show()




