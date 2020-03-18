import sys
import numpy as np
import collections
# 완전탐색
def manhattan_dist(a, b):
    return abs(a[0]-b[0])+np.abs(a[1]-b[1])

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

    average_num = W * H / 10
    num_of_values = [np.count_nonzero(keypad == i) for i in range(10)]
    under_average_num = np.asarray(num_of_values) < average_num

    for num in query:
        if num == keypad[start_y][start_x]:
            pass
        if under_average_num[num]:
            points = np.where(keypad == num)
            for i in range(len(points[0])):
                starty, startx = start_y, start_x
                y, x = points[0][i], points[1][i]
                dist = manhattan_dist((starty, startx), (y, x))
                if nearest_distance[num] > dist:
                    nearest_distance[num] = dist
                visited[y][x] = 1
            find_list[num] = 1


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

