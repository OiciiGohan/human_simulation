import pygame
from pygame.locals import *
import sys
import numpy as np
import math
import os
import copy
import string

'''import chainer
import chainer
import chainer.links as L
import chainer.functions as F
import chainerrl
from chainer import Sequential'''

def base_cvt(value, n=2):
    numbers = "0123456789"
    alphabets = string.ascii_letters # a-z+A-Zをロード
    characters = numbers + alphabets
    characters = '⓪①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳㉑㉒㉓㉔㉕㉖㉗㉘㉙㉚㉛㉜㉝㉞㉟㊱㊲㊳㊴㊵㊶㊷㊸㊹㊺㊻㊼㊽㊾㊿'
    try:
        tmp = int(value)
    except:
        raise ValueError('Invalid value:', value)
    if n < 2 or n > len(characters):
        raise ValueError('Invalid n:', value)
    result = ''
    tmp = int(value)
    while tmp >= n:
        idx = tmp%n
        result = characters[idx] + result
        tmp = int(tmp / n)
    idx = tmp%n
    result = characters[idx] + result
    return result

def auto_create_maps(width, height):
    created_map = []
    for i in range(height):
        temp_map_row = []
        for j in range(width):
            if i == 0 or j == 0 or j == width-1 or i == height-1:
                temp_map_row.append('x')
            else:
                temp_map_row.append(np.random.choice(['x','o','f'], p=[0.1, 0.85, 0.05]))
        created_map.append(temp_map_row)
    return created_map

def define_human_name(min_num=1, max_num=4):
    temp_str = ''
    name_len = np.random.choice(range(min_num, max_num, 1))
    for i in range(name_len):
        vowls = ['a', 'e', 'i', 'o', 'u', 'æ', 'œ', 'ю']
        head_cs = ['', 'b', 'c', 'ch', 'd', 'dh', 'f', 'fh', 'g', 'gh', 'h', 'j', 'k', 'l', 'm', 'n', 'p', 'ph', 'qu', 'r', 's', 'sh', 't', 'th', 'ts',
                    'w', 'y', 'z', 'mh', 'kr', 'tr', 'pr', 'gr', 'dr', 'br', 'pl', 'bl', 'sp', 'st', 'sk',
                    'д', 'ф', 'ж', 'л', 'ш', 'ç', 'rh', 'lh', 'wh', 'kh', 'x', 'zh', 'bh', 'hm', 'nh', 'gn', 'kn', 'gl', 'pt']
        end_cs = ['', 'b', 'ch', 'd', 'f', 'g', 'h', 'k', 'l', 'm', 'n', 'p', 'r', 's', 'sh', 't', 'w', 'y', 'z']
        if len(temp_str) >= 2:
            if not(temp_str[-1] in vowls and temp_str[-2] in vowls) and np.random.randint(0,1) == 0:
                head_cs = ['']
            if temp_str[-1] in ['b','d','g','p','t','k','ch','f']:
                head_cs = ['', 'r', 's', 'l']
        head_c = np.random.choice(head_cs)
        temp_str += head_c
        if head_c == 'qu':
            vowl = np.random.choice(['a', 'e', 'i', 'o'])
        else:
            vowl = np.random.choice(vowls)
        temp_str += vowl
        if i == name_len - 1:
            last_cs_list = ['x', 'ch', 'ng', 'gh', 'ts', 'ff', 'rh', 'sk', 'st', 'q', 'th', 'tz']
            end_cs.extend(last_cs_list)
        if np.random.randint(0,2) == 0:
            end_cs = ['']
        end_c = np.random.choice(end_cs)
        temp_str += end_c
        temp_str = temp_str[0].upper() + temp_str[1:]
    #print(temp_str)
    return temp_str

def create_human(maps, number, prev_human=[]):
    humans = []
    if prev_human != []:
        humans.extend(prev_human)
    for i in range(number):
        one_human = {}
        one_human['id'] = i + 1
        one_human['y'] = np.random.randint(0, len(maps)-1)
        one_human['x'] = np.random.randint(0, len(maps[0])-1)

        if humans != []:
            for j in range(len(humans)):
                while maps[one_human['y']][one_human['x']] == 'x' or \
                    (one_human['id'] != humans[j]['id'] and one_human['y'] == humans[j]['y'] and one_human['x'] == humans[j]['x']):
                    one_human['y'] = np.random.randint(0, len(maps)-1)
                    one_human['x'] = np.random.randint(0, len(maps[0])-1)
        else:
            while maps[one_human['y']][one_human['x']] == 'x':
                one_human['y'] = np.random.randint(0, len(maps)-1)
                one_human['x'] = np.random.randint(0, len(maps[0])-1)
        one_human['direction'] = np.random.randint(0, 3) #0:上, 1:右, 2:下, 3:左
        one_human['name'] = define_human_name()
        one_human['next_action'] = 'None'
        one_human['next_action_option'] = np.random.randint(0,1000)
        one_human['p_actions'] = [0]
        one_human['sex'] = np.random.choice(['m', 'f'])
        one_human['age'] = np.random.randint(10,60)
        if one_human['sex'] == 'm':
            one_human['max_HP'] = int(np.random.normal(120,15) * (1 - 1 / (math.exp(-abs(one_human['age'] - 15)/100) + 1)) * 2)
        elif one_human['sex'] == 'f':
            one_human['max_HP'] = int(np.random.normal(80,10) * (1 - 1 / (math.exp(-abs(one_human['age'] - 13)/100) + 1)) * 2)
        one_human['HP'] = one_human['max_HP']
        one_human['max_F/H'] = int(np.random.normal(100,10))
        one_human['F/H'] = one_human['max_F/H']
        one_human['status'] = []    #状態異常
        one_human['arm_num'] = int(np.random.normal(2.5,0.2))
        one_human['in_hand'] = np.random.choice([{'item':'None','type':'?'}], size=one_human['arm_num'])
        one_human['burnt_level'] = 0
        one_human['happiness'] = 100
        one_human['lifetime'] = 0
        one_human['vision'] =[[['?',{'item':'?','type':'?'},{'id':'?','sex':'?','age':'?'}] for i in range(3)] for i in range(2)]
        one_human['stopping_time'] = 0
        one_human['sleepiness'] = 0
        one_human['max_sleepiness'] = int(np.random.normal(24,3))
        one_human['sleeping_time'] = 0
        one_human['fear_to_fire'] = np.random.randint(0,100)    #この値が大きいほど火に近づかなくなる
        one_human['sensitivity_to_hunger'] = np.random.randint(40,80)   #この値が大きいほど空腹になりやすい。
        one_human['selfishness'] = np.random.randint(0,100)     #この値が大きいほど利己的な行動をする
        one_human['skin_color'] = np.random.randint(0,17)

        #入力はHP, 満腹度, 火傷の度合い, 幸福度, 今いる場所に火があるかどうか, 目の前に火が(ry…（前横斜めの5方向）, 
        # 今アイテム、前アイテム…、目の前,横,斜め前に壁があるか、手持ちアイテム（腕の数により変動）、
        #　手持ちアイテムの経過時間、
        '''n_input = 2
        n_hidden = [3]
        n_output = 1
        one_human['net'] = Sequential(L.Linear(n_input, n_hidden[0]), F.sigmoid)
        for i in range(len(n_hidden)):
            if i != len(n_hidden) - 1:
                one_human['net'] += Sequential(L.Linear(n_hidden[i], n_hidden[i + 1]), F.sigmoid)
        one_human['net'] += Sequential(L.Linear(n_hidden[-1], n_output))
        optimizer = chainer.optimizers.SGD(lr=0.5)
        optimizer.setup(net)'''

        humans.append(one_human)
    return humans

def create_items(maps):
    items = copy.deepcopy(maps)
    item_id = 0
    global items_dict
    items_dict = [{'No.':0, 'item':'None', 'type':'None'}, 
                  {'No.':1, 'item':'apple', 'state':'normal', 'time':0, 'type':'food', '+F/H':20, '+HP':10, '+happiness':20}]
    for y in range(len(maps)):
        for x in range(len(maps[y])):
            if maps[y][x] in ['x', 'f']:
                items[y][x] = copy.deepcopy(items_dict[0])
            else:
                item_type = np.random.choice(['None','apple'],p=[0.0, 1.0])
                if item_type == 'None':
                    items[y][x] = copy.deepcopy(items_dict[0])
                else:
                    items[y][x] = copy.deepcopy(items_dict[np.random.choice([0, 1], p=[0.0, 1.0])])
                    items[y][x]['id'] = item_id
                    item_id += 1
    global max_item_id
    max_item_id = item_id
    return items



def render_maps(screen, maps, size, relief):
    for y in range(len(maps)):
        for x in range(len(maps[y])):
            if maps[y][x] == 'x':
                pygame.draw.rect(screen,(0,0,0),Rect(x * size + relief, y * size + relief, size, size))
            elif maps[y][x] == 'o':
                pygame.draw.rect(screen,(255,255,255),Rect(x * size + relief, y * size + relief, size, size))
            elif maps[y][x] == 'f':
                pygame.draw.rect(screen,(255,0,0),Rect(x * size + relief, y * size + relief, size, size))

def human_who(one_human, other_human):
    who = {}
    who['id'] = other_human['id']
    who['sex'] = other_human['sex']
    if other_human['age'] < 5:
        who['age'] = 'baby'
    elif other_human['age'] < 18:
        who['age'] = 'child'
    elif other_human['age'] < 30:
        who['age'] = 'young'
    elif other_human['age'] < 45:
        who['age'] = 'adult'
    elif other_human['age'] < 55:
        who['age'] = 'middle'
    elif other_human['age'] < 65:
        who['age'] = 'mature'
    else:
        who['age'] = 'old'
    if other_human['HP'] < other_human['max_HP'] * 0.1:
        who['health'] = 'very_bad'
    elif other_human['HP'] < other_human['max_HP'] * 0.4:
        who['health'] = 'bad'
    elif other_human['HP'] < other_human['max_HP'] * 0.7:
        who['health'] = 'normal'
    else:
        who['health'] = 'good'
    return who

def human_vision(one_human, maps, items, humans):
    if 'dead' in one_human['status']:
        one_human['vision'] = [[['?',{'item':'?','type':'?'},{'id':'?','sex':'?','age':'?'}] for i in range(3)] for i in range(2)]
    else:
        one_human['vision'][1][1][0] = maps[one_human['y']][one_human['x']]
        one_human['vision'][1][1][1] = items[one_human['y']][one_human['x']]
        neighbor_human = [[{'id':'?','sex':'?','age':'?'}for i in range(3)] for i in range(3)]
        for i in range(len(humans)):
            if humans[i]['id'] != one_human['id']:
                if humans[i]['y'] == one_human['y'] - 1 and humans[i]['x'] == one_human['x'] - 1:
                    neighbor_human[0][0] = human_who(one_human, humans[i])
                if humans[i]['y'] == one_human['y'] - 1 and humans[i]['x'] == one_human['x']:
                    neighbor_human[0][1] = human_who(one_human, humans[i])
                if humans[i]['y'] == one_human['y'] - 1 and humans[i]['x'] == one_human['x'] + 1:
                    neighbor_human[0][2] = human_who(one_human, humans[i])
                if humans[i]['y'] == one_human['y'] and humans[i]['x'] == one_human['x'] - 1:
                    neighbor_human[1][0] = human_who(one_human, humans[i])
                if humans[i]['y'] == one_human['y'] and humans[i]['x'] == one_human['x']:
                    neighbor_human[1][1] = human_who(one_human, humans[i])
                if humans[i]['y'] == one_human['y'] and humans[i]['x'] == one_human['x'] + 1:
                    neighbor_human[1][2] = human_who(one_human, humans[i])
                if humans[i]['y'] == one_human['y'] + 1 and humans[i]['x'] == one_human['x'] - 1:
                    neighbor_human[2][0] = human_who(one_human, humans[i])
                if humans[i]['y'] == one_human['y'] + 1 and humans[i]['x'] == one_human['x']:
                    neighbor_human[2][1] = human_who(one_human, humans[i])
                if humans[i]['y'] == one_human['y'] + 1 and humans[i]['x'] == one_human['x'] + 1:
                    neighbor_human[2][2] = human_who(one_human, humans[i])
        one_human['vision'][1][1][2] = copy.deepcopy(neighbor_human[1][1])
        if one_human['direction'] == 0:
            one_human['vision'][0][0][0] = maps[one_human['y'] - 1][one_human['x'] - 1]     #人間から見て左前
            one_human['vision'][0][0][1] = copy.deepcopy(items[one_human['y'] - 1][one_human['x'] - 1])
            one_human['vision'][0][0][2] = copy.deepcopy(neighbor_human[0][0])
            one_human['vision'][0][1][0] = maps[one_human['y'] - 1][one_human['x']]      #人間から見て前
            one_human['vision'][0][1][1] = copy.deepcopy(items[one_human['y'] - 1][one_human['x']])
            one_human['vision'][0][0][2] = copy.deepcopy(neighbor_human[0][1])
            one_human['vision'][0][2][0] = maps[one_human['y'] - 1][one_human['x'] + 1]     #人間から見て右前
            one_human['vision'][0][2][1] = copy.deepcopy(items[one_human['y'] - 1][one_human['x'] + 1])
            one_human['vision'][0][0][2] = copy.deepcopy(neighbor_human[0][2])
            one_human['vision'][1][0][0] = maps[one_human['y']][one_human['x'] - 1]         #人間から見て左
            one_human['vision'][1][0][1] = copy.deepcopy(items[one_human['y']][one_human['x'] - 1])
            one_human['vision'][0][0][2] = copy.deepcopy(neighbor_human[1][0])
            one_human['vision'][1][2][0] = maps[one_human['y']][one_human['x'] + 1]         #人間から見て右
            one_human['vision'][1][2][1] = copy.deepcopy(items[one_human['y']][one_human['x'] + 1])
            one_human['vision'][0][0][2] = copy.deepcopy(neighbor_human[1][2])
        elif one_human['direction'] == 1:
            one_human['vision'][0][0][0] = maps[one_human['y'] - 1][one_human['x'] + 1]     #人間から見て左前
            one_human['vision'][0][0][1] = copy.deepcopy(items[one_human['y'] - 1][one_human['x'] + 1])
            one_human['vision'][0][0][2] = copy.deepcopy(neighbor_human[0][2])
            one_human['vision'][0][1][0] = maps[one_human['y']][one_human['x'] + 1]      #人間から見て前
            one_human['vision'][0][1][1] = copy.deepcopy(items[one_human['y']][one_human['x'] + 1])
            one_human['vision'][0][1][2] = copy.deepcopy(neighbor_human[1][2])
            one_human['vision'][0][2][0] = maps[one_human['y'] + 1][one_human['x'] + 1]     #人間から見て右前
            one_human['vision'][0][2][1] = copy.deepcopy(items[one_human['y'] + 1][one_human['x'] + 1])
            one_human['vision'][0][2][2] = copy.deepcopy(neighbor_human[2][2])
            one_human['vision'][1][0][0] = maps[one_human['y'] - 1][one_human['x']]         #人間から見て左
            one_human['vision'][1][0][1] = copy.deepcopy(items[one_human['y'] - 1][one_human['x']])
            one_human['vision'][1][0][2] = copy.deepcopy(neighbor_human[0][1])
            one_human['vision'][1][2][0] = maps[one_human['y'] + 1][one_human['x']]         #人間から見て右
            one_human['vision'][1][2][1] = copy.deepcopy(items[one_human['y'] + 1][one_human['x']])
            one_human['vision'][1][2][2] = copy.deepcopy(neighbor_human[2][1])
        elif one_human['direction'] == 2:
            one_human['vision'][0][0][0] = maps[one_human['y'] + 1][one_human['x'] + 1]     #人間から見て左前
            one_human['vision'][0][0][1] = copy.deepcopy(items[one_human['y'] + 1][one_human['x'] + 1])
            one_human['vision'][0][0][2] = copy.deepcopy(neighbor_human[2][2])
            one_human['vision'][0][1][0] = maps[one_human['y'] + 1][one_human['x']]      #人間から見て前
            one_human['vision'][0][1][1] = copy.deepcopy(items[one_human['y'] + 1][one_human['x']])
            one_human['vision'][0][1][2] = copy.deepcopy(neighbor_human[2][1])
            one_human['vision'][0][2][0] = maps[one_human['y'] + 1][one_human['x'] - 1]     #人間から見て右前
            one_human['vision'][0][2][1] = copy.deepcopy(items[one_human['y'] + 1][one_human['x'] - 1])
            one_human['vision'][0][2][2] = copy.deepcopy(neighbor_human[2][0])
            one_human['vision'][1][0][0] = maps[one_human['y']][one_human['x'] + 1]         #人間から見て左
            one_human['vision'][1][0][1] = copy.deepcopy(items[one_human['y']][one_human['x'] + 1])
            one_human['vision'][1][0][2] = copy.deepcopy(neighbor_human[1][2])
            one_human['vision'][1][2][0] = maps[one_human['y']][one_human['x'] - 1]         #人間から見て右
            one_human['vision'][1][2][1] = copy.deepcopy(items[one_human['y']][one_human['x'] - 1])
            one_human['vision'][1][2][2] = copy.deepcopy(neighbor_human[1][0])
        elif one_human['direction'] == 3:
            one_human['vision'][0][0][0] = maps[one_human['y'] + 1][one_human['x'] - 1]     #人間から見て左前
            one_human['vision'][0][0][1] = copy.deepcopy(items[one_human['y'] + 1][one_human['x'] - 1])
            one_human['vision'][0][0][2] = copy.deepcopy(neighbor_human[2][0])
            one_human['vision'][0][1][0] = maps[one_human['y']][one_human['x'] - 1]      #人間から見て前
            one_human['vision'][0][1][1] = copy.deepcopy(items[one_human['y']][one_human['x'] - 1])
            one_human['vision'][0][1][2] = copy.deepcopy(neighbor_human[1][0])
            one_human['vision'][0][2][0] = maps[one_human['y'] - 1][one_human['x'] - 1]     #人間から見て右前
            one_human['vision'][0][2][1] = copy.deepcopy(items[one_human['y'] - 1][one_human['x'] - 1])
            one_human['vision'][0][2][2] = copy.deepcopy(neighbor_human[0][0])
            one_human['vision'][1][0][0] = maps[one_human['y'] + 1][one_human['x']]         #人間から見て左
            one_human['vision'][1][0][1] = copy.deepcopy(items[one_human['y'] + 1][one_human['x']])
            one_human['vision'][1][0][2] = copy.deepcopy(neighbor_human[2][1])
            one_human['vision'][1][2][0] = maps[one_human['y'] - 1][one_human['x']]         #人間から見て右
            one_human['vision'][1][2][1] = copy.deepcopy(items[one_human['y'] - 1][one_human['x']])
            one_human['vision'][1][2][2] = copy.deepcopy(neighbor_human[0][1])
        if one_human['vision'][0][1][0] == 'x' and one_human['vision'][1][0][0] == 'x':
            one_human['vision'][0][0][0] = '?'
            one_human['vision'][0][0][1] = {'item':'?','type':'?'}
            one_human['vision'][0][0][2] = {'id':'?','sex':'?','age':'?'}
        if one_human['vision'][0][1][0] == 'x' and one_human['vision'][1][2][0] == 'x':
            one_human['vision'][0][2][0] = '?'
            one_human['vision'][0][2][1] = {'item':'?','type':'?'}
            one_human['vision'][0][2][2] = {'id':'?','sex':'?','age':'?'}
    return one_human

def action_determine(one_human, elapsed_time):
    action_list = ['None', 'turn_reverse', 'turn_right', 'turn_left', 'walk', 'take', 'eat', 'sleep', 'put']
    #ランダムに行動
    if np.all(one_human['p_actions'] == [0]):
        one_human['p_actions'] = np.ones(len(action_list))
    #一定時間ごとに欲求をリセットする（人間は長時間同じことをしようとしない）
    if elapsed_time % 8:
        one_human['p_actions'] += np.ones(len(action_list))*0.8
    one_human['p_actions'] /= np.sum(one_human['p_actions'])    #確率の正規化
    #ずっと止まっていると動きたくなる(ただし、疲弊していると動けなくなる)
    want_move_threshold = 3
    want_move_constant = math.tanh(one_human['stopping_time']) * 0.1
    if one_human['stopping_time'] >= want_move_threshold:
        one_human['p_actions'][action_list.index('walk')] += want_move_constant * (one_human['HP'] / one_human['max_HP'])
    one_human['p_actions'] /= np.sum(one_human['p_actions'])    #確率の正規化
    #視界を利用してよりよい行動をとる
    #空腹時には食べ物を探して食べることを優先
    if 'hungry' in one_human['status']:
        hunger_constant = (1 - one_human['F/H'] / one_human['max_F/H']) * 10
        food_in_hand = False
        food_hand_index = []
        for i in range(len(one_human['in_hand'])):
            if one_human['in_hand'][i]['item'] != 'None':
                if one_human['in_hand'][i]['type'] == 'food':
                    food_in_hand = True
                    food_hand_index.append(i)
        if food_in_hand:
            one_human['p_actions'][action_list.index('eat')] += hunger_constant
            one_human['next_action_option'] = np.random.choice(food_hand_index)
        elif one_human['vision'][1][1][1]['item'] != 'None':
            if one_human['vision'][1][1][1]['type'] == 'food':
                one_human['p_actions'][action_list.index('take')] += hunger_constant
        elif one_human['vision'][0][1][1]['item'] != 'None':
            if one_human['vision'][0][1][1]['type'] == 'food':
                one_human['p_actions'][action_list.index('walk')] += hunger_constant
        elif one_human['vision'][1][0][1]['item'] != 'None':
            if one_human['vision'][1][0][1]['type'] == 'food':
                one_human['p_actions'][action_list.index('turn_left')] += hunger_constant
        elif one_human['vision'][1][2][1]['item'] != 'None':
            if one_human['vision'][1][2][1]['type'] == 'food':
                one_human['p_actions'][action_list.index('turn_right')] += hunger_constant
        elif one_human['vision'][0][0][1]['item'] != 'None':
            if one_human['vision'][0][0][1]['type'] == 'food':
                one_human['p_actions'][action_list.index('turn_left')] += hunger_constant
                one_human['p_actions'][action_list.index('walk')] += hunger_constant
        elif one_human['vision'][0][2][1]['item'] != 'None':
            if one_human['vision'][0][2][1]['type'] == 'food':
                one_human['p_actions'][action_list.index('turn_right')] += hunger_constant
                one_human['p_actions'][action_list.index('walk')] += hunger_constant
    one_human['p_actions'] /= np.sum(one_human['p_actions'])    #確率の正規化
    #火を避けるように動く。火傷によってダメージを受けるとこの値は増して行くが、火傷しなければどんどん値が元に戻ろうとする
    avoid_fire_constant = one_human['fear_to_fire'] / 100
    if one_human['vision'][1][1][0] == 'f':
        if one_human['vision'][0][1][0] != 'f':
            one_human['p_actions'][action_list.index('walk')] *= avoid_fire_constant
    if one_human['vision'][0][1][0] == 'f':
        one_human['p_actions'][action_list.index('walk')] *= 1 - avoid_fire_constant
        one_human['p_actions'][action_list.index('turn_reverse')] += 0.5 * avoid_fire_constant
        one_human['p_actions'][action_list.index('turn_left')] += 0.5 * avoid_fire_constant
        one_human['p_actions'][action_list.index('turn_right')] += 0.5 * avoid_fire_constant
        if one_human['vision'][1][0][0] == 'f' and one_human['vision'][1][2][0] == 'f':
            one_human['p_actions'][action_list.index('turn_reverse')] += 0.5 * avoid_fire_constant
    one_human['p_actions'] /= np.sum(one_human['p_actions'])    #確率の正規化
    #不可能なことはしようとしない
    cannot_constant = 2
    if one_human['vision'][0][1][0] == 'x':
        one_human['p_actions'][action_list.index('walk')] /= cannot_constant
        one_human['p_actions'][action_list.index('turn_reverse')] += 0.5
        one_human['p_actions'][action_list.index('turn_left')] += 0.5
        one_human['p_actions'][action_list.index('turn_right')] += 0.5
        if one_human['vision'][1][0][0] == 'x' and one_human['vision'][1][2][0] == 'x':
            one_human['p_actions'][action_list.index('turn_reverse')] += 0.5
    food_in_hand = False
    for i in range(len(one_human['in_hand'])):
        if one_human['in_hand'][i]['item'] != 'None':
            if one_human['in_hand'][i]['type'] == 'food':
                food_in_hand = True
    if food_in_hand != True:
        one_human['p_actions'][action_list.index('eat')] /= cannot_constant
    if one_human['vision'][1][1][1]['item'] == 'None':
        one_human['p_actions'][action_list.index('take')] /= cannot_constant
    one_human['p_actions'] /= np.sum(one_human['p_actions'])    #確率の正規化
    #意味なく壁の方を向かない
    dont_see_wall_constant = 2
    if one_human['vision'][1][0][0] == 'x':
        one_human['p_actions'][action_list.index('turn_left')] /= dont_see_wall_constant
    if one_human['vision'][1][2][0] == 'x':
        one_human['p_actions'][action_list.index('turn_right')] /= dont_see_wall_constant
    one_human['p_actions'] /= np.sum(one_human['p_actions'])    #確率の正規化

    one_human['next_action'] = np.random.choice(action_list, p=one_human['p_actions'])
    one_human['next_action_option'] = np.random.randint(0,1000)
    #睡眠の限界が来たら眠る。
    sleep_prob = one_human['sleepiness'] / one_human['max_sleepiness']
    one_human['p_actions'][action_list.index('sleep')] = sleep_prob
    one_human['p_actions'] /= np.sum(one_human['p_actions'])    #確率の正規化
    if one_human['sleepiness'] >= one_human['max_sleepiness']:
        one_human['next_action'] = 'sleep'
    #寝ている間は寝続けるか目覚めるかの2択。眠気がひどいほど目覚めづらくなる.
    if 'sleeping' in one_human['status']:
        wakeup_prob = 1 - one_human['sleepiness'] / one_human['max_sleepiness']
        one_human['next_action'] = np.random.choice(['sleep', 'wake'], p=[1-wakeup_prob, wakeup_prob])
    #死んだら何もできない
    if 'dead' in one_human['status']:
        one_human['next_action'] = 'None'
    return one_human

def action_execute(one_human, maps, humans, items):
    if one_human['next_action'] == 'None':
        if one_human['F/H'] != 0 and one_human['HP'] != 0 and 'dead' not in one_human['status']:
            one_human['stopping_time'] += 1
            one_human['HP'] += 1
            if one_human['HP'] > one_human['max_HP']:
                    one_human['HP'] = one_human['max_HP']
    elif one_human['next_action'] == 'turn_reverse':
        one_human['stopping_time'] += 1
        one_human['direction'] = (one_human['direction'] + 2) % 4
    elif one_human['next_action'] == 'turn_right':
        one_human['stopping_time'] += 1
        one_human['direction'] = (one_human['direction'] + 1) % 4
    elif one_human['next_action'] == 'turn_left':
        one_human['stopping_time'] += 1
        one_human['direction'] = (one_human['direction'] + 3) % 4
    elif one_human['next_action'] == 'walk':
        if one_human['direction'] == 0 and maps[one_human['y'] - 1][one_human['x']] != 'x':
            one_human['y'] -= 1
            one_human['stopping_time'] = 0
        elif one_human['direction'] == 1 and maps[one_human['y']][one_human['x'] + 1] != 'x':
            one_human['x'] += 1
            one_human['stopping_time'] = 0
        elif one_human['direction'] == 2 and maps[one_human['y'] + 1][one_human['x']] != 'x':
            one_human['y'] += 1
            one_human['stopping_time'] = 0
        elif one_human['direction'] == 3 and maps[one_human['y']][one_human['x'] - 1] != 'x':
            one_human['x'] -= 1
            one_human['stopping_time'] = 0
    elif one_human['next_action'] == 'take':
        one_human['stopping_time'] += 1
        if items[one_human['y']][one_human['x']]['item'] != 'None' and {'item':'None'} in one_human['in_hand']:
            open_hand = [i for i, x in enumerate(one_human['in_hand']) if x == {'item':'None'}]
            one_human['in_hand'][open_hand[one_human['next_action_option'] % len(open_hand)]] = items[one_human['y']][one_human['x']]
            items[one_human['y']][one_human['x']] = {'item':'None'}
    elif one_human['next_action'] == 'put':
        one_human['stopping_time'] += 1
        if items[one_human['y']][one_human['x']]['item'] == 'None' and any(one_human['in_hand'] != {'item':'None'}):
            having_hand = [i for i, x in enumerate(one_human['in_hand']) if x != {'item':'None'}]
            selected_item_index = one_human['next_action_option'] % len(having_hand)
            items[one_human['y']][one_human['x']] = one_human['in_hand'][having_hand[selected_item_index]]
            one_human['in_hand'][having_hand[selected_item_index]] = {'item':'None'}
    elif one_human['next_action'] == 'eat' and 'dead' not in one_human['status']:
        one_human['stopping_time'] += 1
        if np.any(one_human['in_hand'] != {'item':'None'}):
            hand_haivng_item = [i for i, x in enumerate(one_human['in_hand']) if x != {'item':'None'}]
            selected_item = hand_haivng_item[one_human['next_action_option'] % len(hand_haivng_item)]
            eaten = one_human['in_hand'][selected_item]
            one_human['in_hand'][selected_item] = {'item':'None'}
            if eaten['type'] == 'food':
                one_human['F/H'] += eaten['+F/H']
                if one_human['F/H'] > one_human['max_F/H']:
                    one_human['F/H'] = one_human['max_F/H']
                    one_human['next_action'] = 'vomit'  #食べ過ぎると嘔吐してHPが減少。
                one_human['HP'] += eaten['+HP']
                if one_human['HP'] > one_human['max_HP']:
                    one_human['HP'] = one_human['max_HP']
                one_human['happiness'] += eaten['+happiness']
    elif one_human['next_action'] == 'vomit':
        one_human['stopping_time'] += 1
        vomit_pain = 3
        one_human['HP'] -= vomit_pain
        if one_human['HP'] < 0:
            one_human['HP'] = 0
        one_human['happiness'] -= 1
        if one_human['happiness'] < 0:
            one_human['happiness'] = 0
    elif one_human['next_action'] == 'sleep':
        if 'sleeping' not in one_human['status']:
            one_human['status'].append('sleeping')
        one_human['sleepiness'] -= 5
        if one_human['sleepiness'] < 0:
           one_human['sleepiness'] = 0
        one_human['HP'] += 2
        if one_human['HP'] > one_human['max_HP']:
            one_human['HP'] = one_human['max_HP']
    elif one_human['next_action'] == 'wake':
        if 'sleeping' in one_human['status']:
            one_human['status'].remove('sleeping')            
    return one_human, items

def human_hunger(humans):
    hunger_loss = 1 #1時間にどれだけ空腹になるか
    for i in range(len(humans)):
        if humans[i]['F/H'] != 0 and ('dead' not in humans[i]['status']):
            humans[i]['F/H'] -= hunger_loss
            if humans[i]['F/H'] < 0:
                humans[i]['F/H'] = 0
        if humans[i]['F/H'] / humans[i]['max_F/H'] < humans[i]['sensitivity_to_hunger'] / 100:
            if 'hungry' not in humans[i]['status']:
                humans[i]['status'].append('hungry')
            if 'dead' not in humans[i]['status']:
                humans[i]['happiness'] -= 1
            if humans[i]['happiness'] < 0:
                humans[i]['happiness'] = 0
        elif 'hungry' in humans[i]['status']:
            humans[i]['status'].remove('hungry')
        if humans[i]['HP'] != 0 and humans[i]['F/H'] == 0:
            humans[i]['HP'] -= 2
            if humans[i]['HP'] < 0:
                humans[i]['HP'] = 0
            if 'dead' not in humans[i]['status']:
                humans[i]['happiness'] -= 5
            if humans[i]['happiness'] < 0:
                humans[i]['happiness'] = 0
    return humans

def human_sleep(humans):
    for i in range(len(humans)):
        if 'dead' not in humans[i]['status'] and humans[i]['sleepiness'] < humans[i]['max_sleepiness']:
            humans[i]['sleepiness'] += 1
    return humans

def human_burn(humans, maps):
    burn_damage = 1 #焼けた床にいるとダメージ
    burn_pain = 10
    burnt_threshold = 20
    for i in range(len(humans)):
        if maps[humans[i]['y']][humans[i]['x']] == 'f':
            humans[i]['HP'] -= burn_damage * 2
            humans[i]['burnt_level'] += 5
            if humans[i]['HP'] < 0:
                humans[i]['HP'] = 0
        else:
            humans[i]['burnt_level'] -= 1
            if humans[i]['burnt_level'] < 0:
                humans[i]['burnt_level'] = 0
        if humans[i]['burnt_level'] >= burnt_threshold:
            humans[i]['HP'] -= burn_damage
            if humans[i]['HP'] < 0:
                humans[i]['HP'] = 0
            if 'dead' not in humans[i]['status']:
                humans[i]['happiness'] -= burn_pain
            if humans[i]['happiness'] < 0:
                humans[i]['happiness'] = 0
            humans[i]['fear_to_fire'] += burn_pain / 2
            if humans[i]['fear_to_fire'] > 100:
                humans[i]['fear_to_fire'] = 100
            if 'burnt' not in humans[i]['status']:
                humans[i]['default_fear_to_fire'] = humans[i]['fear_to_fire']
                humans[i]['status'].append('burnt')
        elif 'burnt' in humans[i]['status']:
            humans[i]['status'].remove('burnt')
            if humans[i]['fear_to_fire'] > humans[i]['default_fear_to_fire']:
                humans[i]['fear_to_fire'] -= 1
    return humans

def human_death(humans):
    tire_thresold = 20
    for i in range(len(humans)):
        if humans[i]['HP'] <= tire_thresold:
            if 'dead' not in humans[i]['status']:
                humans[i]['happiness'] -= 1
            if humans[i]['happiness'] < 0:
                humans[i]['happiness'] = 0
            if 'tired' not in humans[i]['status']:
                humans[i]['status'].append('tired')
        else:
            if 'tired' in humans[i]['status']:
                humans[i]['status'].remove('tired')
        if humans[i]['HP'] == 0:
            if 'dead' not in humans[i]['status']:
                humans[i]['status'].append('dead')
                #humans[i]['happiness'] = 0
    return humans

def human_reset(humans, maps, items, elapsed_time):
    humans = copy.deepcopy(initial_humans)
    maps = copy.deepcopy(initial_maps)
    items = copy.deepcopy(initial_items)
    elapsed_time = 0
    return humans, maps, items, elapsed_time

def main():
    pygame.init()
    cell_size = 25
    map_w = 30
    map_h = 30
    relief = 20 #余白サイズ
    menu_w = 320
    screen = pygame.display.set_mode((map_w*cell_size + relief*3 + menu_w, map_h*cell_size + relief*2))
    pygame.display.set_caption('game-1')
    #print(pygame.font.get_fonts())  #フォント一覧を調べたいとき
    font1 = pygame.font.SysFont("源ノ角ゴシック", int(cell_size/1.5))
    font1_dead = pygame.font.SysFont("源ノ角ゴシック", int(cell_size/1.5))

    menu_font_size = int(cell_size/1.6)
    font2 = pygame.font.SysFont("hg丸ｺﾞｼｯｸmpro", menu_font_size)
    
    menu_human = 0 #メニューにステータスとして表示する人間
    portrait_w = 128
    portrait_h = 128
    elapsed_time = 0
    arrow_w = 64
    arrow_h = 64
    reset_w = 200
    reset_h = 32

    maps = auto_create_maps(map_w, map_h)
    global initial_maps
    initial_maps = copy.deepcopy(maps)
    humans = create_human(maps, 15)
    global initial_humans
    initial_humans = copy.deepcopy(humans)
    items = create_items(maps)
    global initial_items
    initial_items = copy.deepcopy(items)
    human_image_up = pygame.transform.scale(pygame.image.load('./pictures/person.png').convert_alpha(), (cell_size, cell_size))
    human_image_right = pygame.transform.scale(pygame.image.load('./pictures/person2.png').convert_alpha(), (cell_size, cell_size))
    human_image_down = pygame.transform.scale(pygame.image.load('./pictures/person3.png').convert_alpha(), (cell_size, cell_size))
    human_image_left = pygame.transform.scale(pygame.image.load('./pictures/person4.png').convert_alpha(), (cell_size, cell_size))
    dead_image_up = pygame.transform.scale(pygame.image.load('./pictures/dead_up.png').convert_alpha(), (cell_size, cell_size))
    dead_image_right = pygame.transform.scale(pygame.image.load('./pictures/dead_right.png').convert_alpha(), (cell_size, cell_size))
    dead_image_down = pygame.transform.scale(pygame.image.load('./pictures/dead_down.png').convert_alpha(), (cell_size, cell_size))
    dead_image_left = pygame.transform.scale(pygame.image.load('./pictures/dead_left.png').convert_alpha(), (cell_size, cell_size))
    apple_image = pygame.transform.scale(pygame.image.load('./pictures/apple.png').convert_alpha(), (cell_size, cell_size))
    next_arrow_image = pygame.transform.scale(pygame.image.load('./pictures/next_human.png').convert_alpha(), (arrow_w, arrow_h))
    prev_arrow_image = pygame.transform.scale(pygame.image.load('./pictures/prev_human.png').convert_alpha(), (arrow_w, arrow_h))

    while True:
        screen.fill((255, 255, 255))
        render_maps(screen, maps, cell_size, relief)

        #アイテムの描画
        for y in range(len(items)):
            for x in range(len(items[y])):
                if items[y][x]['item'] == 'apple':
                    screen.blit(apple_image, (x*cell_size + relief, y*cell_size + relief))
        #人間の描画
        all_humans_death_flag = 0
        humans = human_hunger(humans)
        humans = human_sleep(humans)
        humans = human_burn(humans, maps)
        humans = human_death(humans)
        for i in range(len(humans)):
            humans[i] = human_vision(humans[i], maps, items, humans)
            id_str_on_map = base_cvt(str(humans[i]['id']),n=50)
            if 'dead' in humans[i]['status']:
                all_humans_death_flag += 1
                text = font1_dead.render(id_str_on_map, True, (0,0,0))
            else:
                text = font1.render(id_str_on_map, True, (0,0,0))
            humans[i], items = action_execute(humans[i], maps, humans, items)
            if humans[i]['direction'] == 0:
                if 'dead' in humans[i]['status']:
                    screen.blit(dead_image_up, (humans[i]['x']*cell_size + relief, humans[i]['y']*cell_size + relief))
                else:
                    screen.blit(human_image_up, (humans[i]['x']*cell_size + relief, humans[i]['y']*cell_size + relief))
            elif humans[i]['direction'] == 1:
                if 'dead' in humans[i]['status']:
                    screen.blit(dead_image_right, (humans[i]['x']*cell_size + relief, humans[i]['y']*cell_size + relief))
                else:
                    screen.blit(human_image_right, (humans[i]['x']*cell_size + relief, humans[i]['y']*cell_size + relief))
            elif humans[i]['direction'] == 2:
                if 'dead' in humans[i]['status']:
                    screen.blit(dead_image_down, (humans[i]['x']*cell_size + relief, humans[i]['y']*cell_size + relief))
                else:
                    screen.blit(human_image_down, (humans[i]['x']*cell_size + relief, humans[i]['y']*cell_size + relief))
            elif humans[i]['direction'] == 3:
                if 'dead' in humans[i]['status']:
                    screen.blit(dead_image_left, (humans[i]['x']*cell_size + relief, humans[i]['y']*cell_size + relief))
                else:
                    screen.blit(human_image_left, (humans[i]['x']*cell_size + relief, humans[i]['y']*cell_size + relief))
            if 'dead' in humans[i]['status']:
                pygame.draw.rect(screen,(255,255,255),Rect((humans[i]['x'] + 1.0)*cell_size, (humans[i]['y'] + 1)*cell_size, cell_size*0.6, cell_size*0.6))
            screen.blit(text, ((humans[i]['x'] + 0.23)*cell_size + relief, (humans[i]['y'] + 0)*cell_size + relief))
            humans[i] = action_determine(humans[i], elapsed_time)
        
        #メニューの描画
        render_y = relief
        pygame.draw.rect(screen,(0,0,0),Rect(map_w*cell_size + relief*2 + menu_w / 2 - portrait_w/2, render_y, portrait_w, portrait_h))
        next_button_rect = pygame.Rect((map_w*cell_size + menu_w / 2 + portrait_w, render_y + portrait_h / 2 - relief), next_arrow_image.get_rect().size)
        prev_button_rect = pygame.Rect((map_w*cell_size + menu_w / 2 - portrait_w + relief, render_y + portrait_h / 2 - relief), prev_arrow_image.get_rect().size)
        if menu_human != len(humans) - 1:
            screen.blit(next_arrow_image, (map_w*cell_size + menu_w / 2 + portrait_w, render_y + portrait_h / 2 - relief))
        if menu_human != 0:
            screen.blit(prev_arrow_image, (map_w*cell_size + menu_w / 2 - portrait_w + relief, render_y + portrait_h / 2 - relief))
        render_y += portrait_h + relief
        text_human_id = font2.render('ID: {}'.format(humans[menu_human]['id']), True, (0,0,0))
        screen.blit(text_human_id, (map_w*cell_size + relief*2, render_y))
        text_human_name = font2.render('名前: {}'.format(humans[menu_human]['name']), True, (0,0,0))
        screen.blit(text_human_name, (map_w*cell_size + relief + menu_w / 2, render_y))
        render_y += menu_font_size + relief / 2
        text_human_sex = font2.render('性別: {}'.format({'m':'男','f':'女'}[humans[menu_human]['sex']]), True, (0,0,0))
        screen.blit(text_human_sex, (map_w*cell_size + relief*2, render_y))
        text_human_age = font2.render('年齢: {}'.format(humans[menu_human]['age']), True, (0,0,0))
        screen.blit(text_human_age, (map_w*cell_size + relief + menu_w / 2, render_y))
        render_y += menu_font_size + relief / 2
        text_human_HP = font2.render('HP: {0}/{1}'.format(humans[menu_human]['HP'], humans[menu_human]['max_HP']), True, (0,0,0))
        pygame.draw.rect(screen,(0,255,0),Rect(map_w*cell_size + relief*2, render_y + menu_font_size + 1, 
                        (menu_w - relief *2) * humans[menu_human]['HP'] / humans[menu_human]['max_HP'], 10))
        pygame.draw.rect(screen,(0,0,0),Rect(map_w*cell_size + relief*2, render_y + menu_font_size + 1, menu_w - relief *2, 10),2)
        screen.blit(text_human_HP, (map_w*cell_size + relief*2, render_y))
        render_y += menu_font_size + relief
        text_human_FH = font2.render('満腹度: {0}/{1}'.format(humans[menu_human]['F/H'], humans[menu_human]['max_F/H']), True, (0,0,0))
        pygame.draw.rect(screen,(0,255,0),Rect(map_w*cell_size + relief*2, render_y + menu_font_size + 1, 
                        (menu_w - relief *2) * humans[menu_human]['F/H'] / humans[menu_human]['max_F/H'], 10))
        pygame.draw.rect(screen,(0,0,0),Rect(map_w*cell_size + relief*2, render_y + menu_font_size + 1, menu_w - relief *2, 10),2)
        screen.blit(text_human_FH, (map_w*cell_size + relief*2, render_y))
        render_y += menu_font_size + relief
        text_human_sleepiness = font2.render('眠気: {0}/{1}'.format(humans[menu_human]['sleepiness'], humans[menu_human]['max_sleepiness']), True, (0,0,0))
        pygame.draw.rect(screen,(255,0,0),Rect(map_w*cell_size + relief*2, render_y + menu_font_size + 1, 
                        (menu_w - relief *2) * humans[menu_human]['sleepiness'] / humans[menu_human]['max_sleepiness'], 10))
        pygame.draw.rect(screen,(0,0,0),Rect(map_w*cell_size + relief*2, render_y + menu_font_size + 1, menu_w - relief *2, 10),2)
        screen.blit(text_human_sleepiness, (map_w*cell_size + relief*2, render_y))
        render_y += menu_font_size + relief
        text_human_sensitivity_to_hunger = font2.render('空腹感度: {}'.format(humans[menu_human]['sensitivity_to_hunger']), True, (0,0,0))
        screen.blit(text_human_sensitivity_to_hunger, (map_w*cell_size + relief*2, render_y))
        text_human_fear_to_fire = font2.render('火への恐怖: {:.0f}'.format(humans[menu_human]['fear_to_fire']), True, (0,0,0))
        screen.blit(text_human_fear_to_fire, (map_w*cell_size + relief*2 + menu_w/2, render_y))
        render_y += menu_font_size + relief / 2
        text_human_happiness = font2.render('幸福度: {}'.format(humans[menu_human]['happiness']), True, (0,0,0))
        screen.blit(text_human_happiness, (map_w*cell_size + relief*2, render_y))
        render_y += menu_font_size + relief / 2
        text_human_position = font2.render('ｘ: {0:2}　ｙ: {1:2}　{2}'.format(humans[menu_human]['x'], humans[menu_human]['y'], 
                                            ['北','東','南','西'][humans[menu_human]['direction']]), True, (0,0,0))
        screen.blit(text_human_position, (map_w*cell_size + relief*2, render_y))
        render_y += menu_font_size + relief / 2
        str_status = ''
        temp_commas = len(humans[menu_human]['status'])
        if temp_commas == 0:
            str_status = 'なし'
        if 'hungry' in humans[menu_human]['status']:
            str_status += '空腹'
            temp_commas -= 1
            if temp_commas != 0:
                str_status += '、'
        if 'burnt' in humans[menu_human]['status']:
            str_status += '火傷'
            temp_commas -= 1
            if temp_commas != 0:
                str_status += '、'
        if 'tired' in humans[menu_human]['status']:
            str_status += '疲労'
            temp_commas -= 1
            if temp_commas != 0:
                str_status += '、'
        if 'sleeping' in humans[menu_human]['status']:
            str_status += '睡眠'
            temp_commas -= 1
            if temp_commas != 0:
                str_status += '、'
        if 'dead' in humans[menu_human]['status']:
            str_status = '死亡'
        text_human_status = font2.render('状態異常: {}'.format(str_status), True, (0,0,0))
        screen.blit(text_human_status, (map_w*cell_size + relief*2, render_y))
        render_y += menu_font_size + relief / 2
        str_in_hand = ''
        for i in range(len(humans[menu_human]['in_hand'])):
            if humans[menu_human]['in_hand'][i]['item'] == 'None':
                str_in_hand += '-'
            elif humans[menu_human]['in_hand'][i]['item'] == 'apple':
                str_in_hand += 'リンゴ'
            if i != len(humans[menu_human]['in_hand']) - 1:
                str_in_hand += '、'
        text_human_in_hand = font2.render('手持ち: {}'.format(str_in_hand), True, (0,0,0))
        screen.blit(text_human_in_hand, (map_w*cell_size + relief*2, render_y))
        render_y += menu_font_size + relief / 2
        if humans[menu_human]['next_action'] == 'None':
            str_next_action = '何もしない'
        elif humans[menu_human]['next_action'] == 'turn_reverse':
            str_next_action = '振り向く'
        elif humans[menu_human]['next_action'] == 'turn_right':
            str_next_action = '右を向く'
        elif humans[menu_human]['next_action'] == 'turn_left':
            str_next_action = '左を向く'
        elif humans[menu_human]['next_action'] == 'walk':
            str_next_action = '前に進む'
        elif humans[menu_human]['next_action'] == 'take':
            str_next_action = 'アイテムを拾う'
        elif humans[menu_human]['next_action'] == 'put':
            str_next_action = 'アイテムを置く'            
        elif humans[menu_human]['next_action'] == 'eat':
            str_next_action = 'アイテムを食べる'
        elif humans[menu_human]['next_action'] == 'vomit':
            str_next_action = '嘔吐'
        elif humans[menu_human]['next_action'] == 'sleep':
            str_next_action = '眠る'            
        text_human_next_action = font2.render('次に取る行動: {}'.format(str_next_action), True, (0,0,0))
        screen.blit(text_human_next_action, (map_w*cell_size + relief*2, render_y))
        render_y += menu_font_size + relief / 2
        text_lifetime = font2.render('生存時間: {0}日{1}時間'.format(int(humans[menu_human]['lifetime']/24), humans[menu_human]['lifetime']%24), True, (0,0,0))
        screen.blit(text_lifetime, (map_w*cell_size + relief*2, render_y))
        render_y += menu_font_size + relief / 2
        text_elapsed_time = font2.render('経過時間: {0}日{1}時間'.format(int(elapsed_time/24), elapsed_time%24), True, (0,0,0))
        screen.blit(text_elapsed_time, (map_w*cell_size + relief*2, render_y))
        render_y += menu_font_size + relief / 2
        #リセットボタン
        reset_button = pygame.Rect(map_w*cell_size + relief*2 + menu_w / 2 - reset_w/2, map_h*cell_size - relief, reset_w, reset_h)
        pygame.draw.rect(screen, (0, 0, 0), reset_button, 2)
        text_reset = font2.render('蘇生させてリセットする', True, (0,0,0))
        screen.blit(text_reset, (map_w*cell_size + relief*2.5 + menu_w / 2 - reset_w/2, map_h*cell_size - relief*0.6))

        #経過時間による処理
        if all_humans_death_flag != len(humans):
            elapsed_time += 1
            for i in range(len(humans)):
                if 'dead' not in humans[i]['status']:
                    humans[i]['lifetime'] += 1
                    if elapsed_time % 24:
                        humans[i]['happiness'] += 1
        
        pygame.display.update()
        pygame.time.wait(100)

        for event in pygame.event.get():
            if event.type == MOUSEBUTTONDOWN:
                if menu_human != len(humans) - 1:
                    if next_button_rect.collidepoint(event.pos):
                        menu_human += 1
                if menu_human != 0:
                    if prev_button_rect.collidepoint(event.pos):
                        menu_human -= 1
                if reset_button.collidepoint(event.pos):
                    humans, maps, items, elapsed_time = human_reset(humans, maps, items, elapsed_time)
            if event.type == QUIT:
                pygame.quit()
                sys.exit()

if __name__ == "__main__":
    main()