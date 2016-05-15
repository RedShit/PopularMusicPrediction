# coding=utf-8
'''
Created on 2016.5.2
craete alibaba Popular Music Prediction DataSet 
@author: sspa
'''
import csv
import re
from __builtin__ import file

import time
def datetime_timestamp(dt):
    if dt < '19970101':
        dt = '19970101'
    dt = dt + ' 00:00:00'
    time.strptime(dt, '%Y%m%d %H:%M:%S')
    s = time.mktime(time.strptime(dt, '%Y%m%d %H:%M:%S'))
    return int(s)
def load_data(load_file_path, query_action_type = '1'):
    print "Load data", load_file_path, " query_action_type", query_action_type
    fd_stdin = csv.reader(file(load_file_path, 'rb'))
    res = {}
    if (re.search('actions.csv', load_file_path)):
        for line in fd_stdin:
            #user_id = line[0]
            song_id = line[1]
            gmt_create = int(line[2])/3600/24
            action_type = line[3]
            if(query_action_type != action_type):
                continue
            res.setdefault(song_id, {})
            res[song_id].setdefault(gmt_create, 0)
            res[song_id][gmt_create] = res[song_id][gmt_create]+1;
    if (re.search('songs.csv', load_file_path)):
        for line in fd_stdin:
            song_id = line[0]
            #artist_id = line[1]
            publish_time = datetime_timestamp(line[2])
            #song_init_plays = int(line[3])
            #Language = int(line[4])
            #Gender = line[5]
            res.setdefault(song_id, 'artist_id')
            res[song_id] = publish_time;
    print "down load complete!"
    return res

def get_artist_actions(actions, songs_publish_time):
    res = {}
    for song_id in actions.keys():
        if song_id not in songs_publish_time.keys():
            continue
        tmp_actions = actions[song_id]
        res.setdefault(song_id, {})
        for gmt_create in tmp_actions.keys():
            res[song_id].setdefault(gmt_create, 0)
            if (songs_publish_time[song_id] <= gmt_create):
                res[song_id][gmt_create] = -1
            else:
                res[song_id][gmt_create] = tmp_actions[gmt_create] + res[song_id][gmt_create]
    return res

def get_predict(actions_type, gmt_time, key, \
                            feature_days = 1, predict_days = 60):
    res = []
    res.append(key)
    for x in range(-feature_days, 0):
        actions_type.setdefault(gmt_time+x, 0)
        if(actions_type[x+gmt_time]<0):
            continue
        res.append(actions_type[x+gmt_time])
    for x in range(0, predict_days):
        actions_type.setdefault(gmt_time+x, 0)
        if(actions_type[x+gmt_time]<0):
            continue
        res.append(actions_type[x+gmt_time])
    return [str(i) for i in res]

def create_data_file(train_file_path, test_file_path, data_type, \
                     feature_days = 1, predict_days = 60, gmt_start = 16495, predict_gmt = 16617):
    train_file = csv.writer(file(train_file_path, 'wb'))
    test_file = csv.writer(file(test_file_path, 'wb'))
    for key in data_type.keys():
        actions_type = data_type[key]
        line = []
        line.append(key)
        for gmt_time in range(gmt_start,predict_gmt):
            actions_type.setdefault(gmt_time, 0)
            if(actions_type[gmt_time]<0):
                continue
            line.append(actions_type[gmt_time])
        train_file.writerow([str(i) for i in line])
        line = get_predict(actions_type, predict_gmt, key)
        test_file.writerow(line)
        
if __name__ == '__main__':
    load_actions_file_path = 'E:/ali/mars_tianchi_user_actions.csv'
    load_songs_file_path = 'E:/ali/mars_tianchi_songs.csv'
    actions_type = load_data(load_actions_file_path, query_action_type='1')
    #actions_type = load_data(load_actions_file_path, query_action_type='2')
    #actions_type = load_data(load_actions_file_path, query_action_type='3')
    songs_publish_time = load_data(load_songs_file_path, query_action_type='0')
    data_type = get_artist_actions(actions_type, songs_publish_time)
    #data_type = get_artist_actions(actions_type, songs)
    #data_type = get_artist_actions(actions_type, songs)
    train_file_path = 'E:/ali/mars_tianchi_train_data.csv'
    test_file_path = 'E:/ali/mars_tianchi_test_data.csv'
    create_data_file(train_file_path, test_file_path, data_type)
