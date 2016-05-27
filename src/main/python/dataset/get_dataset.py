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
def load_data(load_file_path, query_action_type = '1'):
    print "Load data", load_file_path, " query_action_type", query_action_type
    fd_stdin = csv.reader(file(load_file_path, 'rb'))
    res = {}
    gmt_max = 0
    if (re.search('actions.csv', load_file_path)):
        for line in fd_stdin:
            #user_id = line[0]
            song_id = line[1]
            gmt_create = int(line[2])/3600/24
            gmt_max = max(gmt_create, gmt_max)
            action_type = line[3]
            if(query_action_type != action_type):
                continue
            res.setdefault(song_id, {})
            res[song_id].setdefault(gmt_create, 0)
            res[song_id][gmt_create] = res[song_id][gmt_create]+1;
    if (re.search('songs.csv', load_file_path)):
        for line in fd_stdin:
            song_id = line[0]
            artist_id = line[1]
            #song_init_plays = int(line[3])
            #Language = int(line[4])
            #Gender = line[5]
            res.setdefault(song_id, 'artist_id')
            res[song_id] = artist_id;
    print "down load complete!"
    return res

def get_artist_actions(actions, songs):
    res = {}
    for song_id in actions.keys():
        if song_id not in songs.keys():
            continue
        tmp_actions = actions[song_id]
        artist_id = songs[song_id]
        res.setdefault(artist_id, {})
        for gmt_create in tmp_actions.keys():
            res[artist_id].setdefault(gmt_create, 0)
            res[artist_id][gmt_create] = tmp_actions[gmt_create] + res[artist_id][gmt_create]
    return res

def create_data_file(dataset_file_path, data_type, gmt_start = 16495, predict_gmt = 16626): #16738, 16626
    dataset_file = csv.writer(file(dataset_file_path, 'wb'))
    for key in data_type.keys():
        actions_type = data_type[key]
        line = []
        line.append(key)
        for gmt_time in range(gmt_start,predict_gmt):
            actions_type.setdefault(gmt_time, 0)
            line.append(actions_type[gmt_time])
        dataset_file.writerow([str(i) for i in line])

def get_artist_id_file(artist_id_file_path, data_type):
    artist_id_file = csv.writer(file(artist_id_file_path, 'wb'))
    for key in data_type.keys():
        artist_id_file.writerow([key])
        
if __name__ == '__main__':
    load_actions_file_path = 'D:/MyEclipse/alibaba/mars_tianchi_user_actions.csv'
    load_songs_file_path = 'D:/MyEclipse/alibaba/mars_tianchi_songs.csv'
    actions_type = load_data(load_actions_file_path, query_action_type='1')
    #actions_type = load_data(load_actions_file_path, query_action_type='2')
    #actions_type = load_data(load_actions_file_path, query_action_type='3')
    songs = load_data(load_songs_file_path, query_action_type='0')
    data_type = get_artist_actions(actions_type, songs)
    #data_type = get_artist_actions(actions_type, songs)
    #data_type = get_artist_actions(actions_type, songs)
    dataset_file_path = 'D:/MyEclipse/alibaba/mars_tianchi_data.csv'
    create_data_file(dataset_file_path, data_type)
    artist_id_file_path = 'D:/MyEclipse/alibaba/mars_tianchi_artist_id.csv'
    get_artist_id_file(artist_id_file_path, data_type)
