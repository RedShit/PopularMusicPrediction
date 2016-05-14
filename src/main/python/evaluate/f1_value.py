import csv
import math
import matplotlib.pyplot as plt

if __name__ == '__main__':
    artist_num = 50
    predict_days = 60
    predict_file_path = '/home/hadoop/aliMatch/data/mars_tianchi_predict_data.csv'
    predict_file = csv.reader(file(predict_file_path, 'rb'))
    predict = []
    for line in predict_file:
        tmp = []
        for x in line:
            tmp.append(float(x))
        predict.append(tmp)
    f1_value = 0.0
    opt_f1_value = 0.0
    for i in range(artist_num):
        index = i*2;
        alpha = 0.0
        N = predict_days
        sum = 0.0
        a = 1.0 # 0.9
        if not predict[index+1][0] == predict[index+1][1]:
            pass
            continue
        X = []
        Y1 = []
        Y2 = []
        for j in range(N):
            X.append(j)
            Y1.append(predict[index][j])
            Y2.append(predict[index+1][j])
        #plt.plot(X, Y1)
        #plt.plot(X, Y2)
        #plt.show()
        for j in range(N):
            S = predict[index+1][j]*a
            T = predict[index][j]
            if T == 0:
                continue
            alpha = alpha + ((S-T)/T)**2
            sum = sum + T
        alpha = math.sqrt(alpha/N)
        f1_value = f1_value + (1.0-alpha)*math.sqrt(sum)
        opt_f1_value = opt_f1_value + math.sqrt(sum)
    print f1_value, opt_f1_value, 100.0*f1_value/opt_f1_value
        
