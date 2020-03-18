'''
map
    0 = 보도 없고 기둥도 없음
    1 = 기둥 있음
    2 = 보 있음
    3 = 둘 다 있음
'''
import numpy as np

def work(map_,x,y,a,b):
    if a == 0:  # 기둥
        if b == 0: # 삭제. 삭제 못하는 케이스?
            if map_[x,y+1] == 1: #위가 기둥일 경우
                if map_[x-1,y+1] == 2 or map_[x-1,y+1] == 3: # 좌측에 보가 있는 경우 삭제
                    if map_[x,y] == 1:
                        map_[x,y] = 0
                    elif map_[x,y] == 3:
                        map_[x,y] = 2
            elif map_[x,y+1] == 2 or map_[x,y+1] == 3: #위에 오른쪽으로 보가 있을 경우(2,3 구분 불필요.)
                if map_[x+1,y+1] == 2 or map_[x+1,y+1] == 3 or map_[x+1,y] == 1 or map_[x+1,y] == 3: # 보의 반대쪽에 기둥이나 다른 보가 있는지 확인.
                    if map_[x, y] == 1:
                        map_[x, y] = 0
                    elif map_[x, y] == 3:
                        map_[x, y] = 2

            elif map_[x-1,y+1] == 2 or map_[x-1,y+1] ==3 : #위에 왼쪽으로 보가 있을 경우
                if map_[x-2,y+1] == 2 or map_[x-2,y+1] == 3 or map_[x-1,y] == 1 or map_[x-1,y] == 3: # 보의 반대쪽에 기둥이나 다른 보가 있는지 확인.
                    if map_[x, y] == 1:
                        map_[x, y] = 0
                    elif map_[x, y] == 3:
                        map_[x, y] = 2

            else : # 해당하지 않으면 그냥 삭제 가능.
                if map_[x, y] == 1:
                    map_[x, y] = 0
                elif map_[x, y] == 3:
                    map_[x, y] = 2

        elif b == 1: # 설치
            if y == 0: # 바닥 일 경우 무조건 설치
                if map_[x,y] == 0:
                    map_[x,y] = 1
                elif map_[x,y] == 2:
                    map_[x, y] = 3
            else:
                if map_[x,y-1] == 1 or map_[x,y-1] ==3 or map_[x-1,y] == 2 or map_[x,y] == 2: # 조건을 만족할 경우 설치
                    if map_[x, y] == 0:
                        map_[x, y] = 1
                    elif map_[x, y] == 2:
                        map_[x, y] = 3



    elif a == 1:  # 보
        if b == 0:  # 삭제. 삭제 못하는 케이스?
            # 좌우에 있는 보가 이 보가 빠질경우 지지기반이 없을 경우
            if map_[x-1,y] == 2 or map_[x-1,y] == 3
            # 좌우에 있는 기둥이 보가 빠질 경우 지지기반이 없는 경우

        elif b == 1:  # 설치
            if map_[x,y-1] == 1 or map_[x,y-1] == 3 or map_[x+1,y-1] == 1 or map_[x+1,y-1] == 3: # 설치하려는 곳 아래에 기둥이 있거나 우측에 기둥이 있을 경우
                if map_[x,y] == 0:
                    map_[x,y] = 2
                elif map_[x,y] == 1:
                    map_[x, y] = 3
            elif (map_[x-1,y] == 2 or map_[x-1,y] == 3) and (map_[x+1,y] == 2 or map_[x+1,y] == 3): # 설치할 때 좌우에 모두 보가 있는 경우
                if map_[x,y] == 0:
                    map_[x,y] = 2
                elif map_[x,y] == 1:
                    map_[x, y] = 3

def solution(n, build_frame):
    answer = [[]]
    map_ = np.zeros([n,n], dtype=int)
    for frame in build_frame:
        x, y, a, b = frame  # x,y 좌표, a=0 기둥 1 보, b= 0 삭제 1 설치


    # [x, y, a] 좌표, 0기둥 1 보
    return answer