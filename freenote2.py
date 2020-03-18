import sys
import numpy as np
import collections
# 완전탐색
def not_finded(find_list, query):
    flag = False
    for num in query:
        if find_list[num] == 0:
            flag = True
    return flag

def solution(keypad, start_x, start_y, query):
    start_x -= 1
    start_y -= 1  # 좌표와 맞추기 위함


    nearest_distance = [sys.maxsize for _ in range(10)]
    find_list = [0 for _ in range(10)]
    visited = np.zeros(keypad.shape)
    four_direc = [(1, 0), (-1,0), (0,1), (0,-1)]

    queue = collections.deque([(start_y, start_x)])
    queue_append = queue.append

    cur_dist = 0
    find_list[keypad[start_y][start_x]] = 1
    visited[start_y][start_x] = 1
    nearest_distance[keypad[start_y][start_x]] = cur_dist

    while len(queue) != 0 and not_finded(find_list, query):
        cur_dist += 1
        len_cur_points = len(queue)
        for _ in range(len_cur_points):
            point = queue.popleft()
            for direc in four_direc:
                y = point[0] + direc[0]
                x = point[1] + direc[1]
                if x <= W-1 and y<=H-1 and x>=0 and y>=0: # 키패드를 넘지 않았을 때
                    if visited[y][x] == 0: # 방문된적이 없다면
                        if find_list[keypad[y][x]] == 0:
                            find_list[keypad[y][x]] = 1 # 새로운 글자를 찾을 시 바뀐다.
                            nearest_distance[keypad[y][x]] = cur_dist # nearest distance에서 해당 글자의 최소치를 갱신한다.
                        visited[y][x] = 1 # 방문 여부 갱신
                        queue_append((y, x)) # 다음 distance에서 계산할 좌표.
    #탐색 완료

    result = []
    for num in query:
        result.append(nearest_distance[num])
    print(' '.join(map(str,result)))

# 입력
W, H = list(map(int,input().split(' ')))
keypad = []
for _ in range(H):
    line = list(map(int, input()))
    keypad.append(line)
keypad = np.asarray(keypad)
Q = int(input())
for _ in range(Q):
    x, y, len_query, query = input().split(' ')
    x, y = int(x), int(y)
    query = list(map(int, list(query)))
    solution(keypad, x, y, query)

'''
5 5
03237
51668
42097
05345
12198
3
3 4 4 3125
1 1 3 513
2 2 2 03

0 1 2 1
1 2 1
2 1

5 5
56047
01715
30483
94291
85489
4
3 3 3 281
4 2 2 45
1 4 2 71
4 3 3 034

1 1 2
1 1
4 3
2 1 1

'''