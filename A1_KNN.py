# Andrew Nash - 201609492
# Yee Teing Lo - 201805462
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from statistics import mode

BONUS = False


def chi2_dist(X, Y):
    dist = 0
    
    for x, y in zip(X, Y):
        dist += ((x - y) ** 2) / (x + y + 0.00001) # .01 prevents divide by 0

    return dist * 0.5


def knn(train, query, k):
    neighbors = []
    
    for i, row in train.iterrows():
        dist = chi2_dist(query.drop('Class').tolist(),
                         row.drop('Class').tolist())
        neighbors.append((dist, row['Class']))
    
    neighbors = sorted(neighbors, key=lambda tup: tup[0])

    neighbors = neighbors[:k]
    predict = mode([row[-1] for row in neighbors]) # need py >= 3.8 for this

    return predict

def train(x_train, x_test, k):
    correct = 0
    for i, row in x_test.iterrows():
        predict = knn(x_train, row, k)
        
        if predict == row['Class']:
            correct += 1
            
    accuracy = correct/len(x_test)*100
    return accuracy
    


def main(file, k, U):
    df = pd.read_csv('Data4A1_NoDup.tsv', sep='\t', header=0)
    x = df.drop(['Sequence.id'], axis=1)
    y = df['Class']
    x_train, x_test, _, _ = train_test_split(x, y, test_size=int(U)/len(df), shuffle=True)
    
    accuracy = train(x_train, x_test, int(k))
    print(f'{k}\t{accuracy}%')
    


if __name__ == "__main__":
    if len(sys.argv) == 4:
        main(sys.argv[1], sys.argv[2], sys.argv[3])
    else:
        main('Data4A1_NoDup.tsv', 5, 50)

    # generate data for bonus
    if BONUS == True:
        df = pd.read_csv('Data4A1_NoDup.tsv', sep='\t', header=0)
        x = df.drop(['Sequence.id'], axis=1)
        y = df['Class']
        
        results = dict()
        for k in range(1, 16):
            avg = 0
            for i in range(0, 10): # 10 random shuffles for better avgs
                x_train, x_test, _, _ = train_test_split(x,y,
                                                         test_size=50/len(df),
                                                         shuffle=True)
                avg += train(x_train, x_test, k)
            
            print(f'{k}\t{avg/10}%')
            results[k] = avg/10

