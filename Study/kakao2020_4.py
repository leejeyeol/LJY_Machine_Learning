def solution(words, queries):
    answer = []
    for querie in queries:
        querie_cnt = 0
        for word in words :
            if len(word) == len(querie):
                flag = True
                for i in range(len(querie)):
                    if querie[i] == '?':
                        pass
                    elif querie[i] != word[i]:
                        flag = False
                if flag == True:
                    querie_cnt += 1
        answer.append(querie_cnt)
    return answer

words = ["frodo", "front", "frost", "frozen", "frame", "kakao"]
queries = ["fro??", "????o", "fr???", "fro???", "pro?"]
print(solution(words,queries))