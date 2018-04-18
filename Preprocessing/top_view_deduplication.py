import scipy.io
import glob
import os
import numpy as np
from numpy.core.records import fromarrays

def load_group_data(file_path):
    a = scipy.io.loadmat(file_path)['Group'][0]
    b = np.ndarray.tolist(a)
    if b[0].dtype == "uint8":
        new_b = []
        for i_b in b:
            if len(i_b) != 0:
                new_b.append(np.ndarray.tolist(i_b[0]))
            else:
                break
        new_b.sort()
        num_of_groups = len(new_b)

    elif b[0].dtype == "object":
        new_b = []
        for i_b in b:

            if i_b[0][0][0][-1] == ' ':
                tmp = i_b[0][0][0][0:-1]
            else:
                tmp = i_b[0][0][0]

            new_b.append(list(map(int, tmp.split(' '))))
        new_b.sort()
        num_of_groups = len(new_b)

    return new_b, num_of_groups


def is_it_over_three(nums_of_groups):
    n = nums_of_groups
    n.sort()
    anchor = None
    max_group = 1
    max = 1
    cur = 1
    for i in n:
        if anchor is None:
            anchor = i
        else:
            if i == anchor:
                cur = cur + 1
            else:
                anchor = i
                if max < cur:
                    max_group = i
                    max = cur
                cur = 1
    if max < cur:
        max_group = i
        max = cur

    if max >= 3:
        return True, max_group
    else:
        return False, max_group

def is_same_group(meta_group, max_group):
    check_bool_list = []
    check_equality = []
    for group in meta_group:
        check_bool_list.append(len(group) == max_group)
    for i in range(max_group):
        check_list = []
        for j in range(len(meta_group)):
            if check_bool_list[j]:
                check_list.append(meta_group[j][i])
        check_equality.append(all(check_list[0] == cl for cl in check_list))
    return all(check_equality)


root_path = "/media/leejeyeol/74B8D3C8B8D38750/Data/Top_View_for_vote" # change to your data root path. "root_path"/hyunil/bus_stop
worker_names = ['hyunil', 'janghak', 'jeongmin', 'jeyeol', 'sungsin']
picture_classification_names = ["bus_stop", "cafeteria", "classroom", "conference", "etc", "library", "park"]

good_case = []
problem_case = []

for picture_classification_name in picture_classification_names:
    print(picture_classification_name)
    file_root_list = []
    for worker_name in worker_names:
        file_root = os.path.join(root_path, worker_name, picture_classification_name)
        file_root_list.append(glob.glob(os.path.join(file_root, "*.mat")))
    for i_file in range(len(file_root_list[0])):
        meta_groups = []
        nums_of_groups = []
        for i_worker in range(len(worker_names)):
            groups, num_of_groups = load_group_data(file_root_list[i_worker][i_file])
            meta_groups.append(groups)
            nums_of_groups.append(num_of_groups)
        yes_or_no, max_group = is_it_over_three(nums_of_groups)
        if yes_or_no:
            good_case.append(os.path.join(picture_classification_name, os.path.basename(file_root_list[i_worker][i_file])))
            if is_same_group(meta_groups, max_group):
                good_case.append(
                    os.path.join(picture_classification_name, os.path.basename(file_root_list[i_worker][i_file])))
            else:
                problem_case.append(
                    os.path.join(picture_classification_name, os.path.basename(file_root_list[i_worker][i_file])))

        else:
            problem_case.append(
                os.path.join(picture_classification_name, os.path.basename(file_root_list[i_worker][i_file])))


'''
test_file_root = os.path.join(root_path,worker_names[0], picture_classification_names[0])
test_files = glob.glob(os.path.join(test_file_root, "*.mat"))

funciton 
a = scipy.io.loadmat(test_files[0])['Group'][0]
b = np.ndarray.tolist(a)
new_b = []
for i_b in b:
    new_b.append(np.ndarray.tolist(i_b[0]))
new_b.sort()
num_of_groups = len(new_b)

# 정렬된 결과 5명의 케이스 에 대하여

# 그룹의 갯수가 3개이상 같을 경우 체크 =>
# \sorting 후 for문으로 다른 숫자 나올떄까지. max같은 값이 3 이상이면 true(함수화)
# true : 같은 그룹의 숫자에 대해서 \중복검사 수행.
# false : problem_case 리스트에 추가. ("bus_stop/010.mat")

# problem_case 저장.
'''

#problem_case = fromarrays(np.asarray(problem_case), names=["file_path"])
#scipy.io.savemat("problem_case_3.mat", mdict={'file_path': problem_case})
np.savetxt("problem_case_3.csv",problem_case, fmt='%s', delimiter=',')
