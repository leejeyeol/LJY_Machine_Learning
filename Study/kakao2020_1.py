def solution(s):
    answer = len(s)
    # 자르는 단위
    for i in range(1, int(len(s) / 2)+1):
        # 이번 단위로 잘랐을 때 압축된 문자열
        part = ""
        part_cnt = 0
        sub_s = ""
        # 이번 단위로 잘랐을 때 각 파트가 시작되는 지점
        for start_point in range(0, len(s), i):
            if part == "":
                part = s[start_point:start_point + i]
                part_cnt += 1
            elif part == s[start_point:start_point + i]:
                part_cnt += 1
            else:  # part와 잘라낸 조각이 다를 경우 잘라낸다.
                if part_cnt == 1:
                    sub_s = sub_s + part
                    part = s[start_point:start_point + i]
                    part_cnt = 1
                else:
                    sub_s = sub_s + str(part_cnt) + part
                    part = s[start_point:start_point + i]
                    part_cnt = 1

        if part != "":  # 아직 subsoltion에 더해지지 않은 채로 종료될 경우
            if part_cnt == 1:
                sub_s = sub_s + part
            else:
                sub_s = sub_s + str(part_cnt) + part


        if answer > len(sub_s):
            answer = len(sub_s)
        print(i)
        print(sub_s)
        print(len(sub_s))
    return answer

print(solution("ababcdcdababcdcd"))