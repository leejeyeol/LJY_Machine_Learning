import re
import math

train_log_file = r'D:\experiments\HANON\log_cop.txt'
transformed_train_log_file = r'D:\experiments\HANON\t_log_train.txt'
transformed_eval_log_file = r'D:\experiments\HANON\t_log_eval.txt'

def remove_char(line):
    return re.sub('\[|\]', '', line)

def time2str(sec):
    h = sec / 3600
    ms = sec % 3600
    m = ms / 60
    s = ms % 60
    micro_sec = s - math.floor(s)
    return "%02d:%02d:%02d ms %.2f" % (h, m, s, micro_sec)

f = open(train_log_file, 'r')
lines = f.read().splitlines()
f.close()

is_train = True
train_lines_out = []
eval_lines_out = []

for l in lines:
    split_e = l.split()
    if l.startswith('Episode number'):
        if l.split()[4] == 'Training':
            is_train = True
        else:
            is_train = False

        train_lines_out.append(l+'\n') if is_train else eval_lines_out.append(l+'\n')


    elif l.startswith("Step number"):
        w_list = l.split(',')

        step_num = int(w_list[0].split()[-1])

        prev_state = remove_char(w_list[1]).split()[2:6]
        prev_state = list(map(float, prev_state))

        action = remove_char(w_list[2]).split()[1:5]
        action = list(map(float, action))

        next_state = remove_char(w_list[3]).split()[2:6]
        next_state = list(map(float, next_state))

        num_of_info = len(w_list)-5
        info_dict = {}
        for i in range(1, num_of_info+1):
            info_dict[w_list[i+3].split()[0]] = float(w_list[i+3].split()[2])

        #cop = float(w_list[4].split()[2])

        sample_list = [format(step_num, '02d'),
                       format(prev_state[0], '.2f'), format(prev_state[1], '.2f'),
                       format(prev_state[2], '.2f'), format(prev_state[3] * 8600, '.2f'),
                       format(next_state[-1] * 8600, '.3f'), format((action[1] * 1500) + 1500, '.3f'),
                       format((action[2] * 0.75) + 1.25, '.3f'), format((action[3] * 0.75) + 1.25, '.3f'),
                       format(next_state[0], '.2f'), format(next_state[1], '.2f'),
                       format(next_state[2], '.2f'), format(next_state[3] * 8600, '.2f'),
                       ]
        info_key_list = [key for key in info_dict.keys()]
        info_value_list = [float(info_dict[key]) for key in info_dict.keys()]
        l_new_info = ''
        for i in range(num_of_info):
             info_sample_list = [format(info_key_list[i], 's'), format(info_value_list[i], '.2f')]
             l_new_info = l_new_info + '{} : {}, '.format(*info_sample_list)


        l_new = 'Step number: {}, Prev state: [{} {} {} {}], Action: [{} {} {} {}], Next state: [{} {} {} {}], '.\
                format(*sample_list)
        l_new = l_new + l_new_info
        train_lines_out.append(l_new+'\n') if is_train else eval_lines_out.append(l_new+'\n')

    elif l.startswith('Episode score'):
        score = float(l.split(',')[0].split()[-1])
        time = float(l.split(',')[1].split()[-1])

        sample_list = [
            format(score, '.2f'),
            format(time2str(time), 's')
        ]

        l_new = 'Episode score: {}, Running time: {}'.format(*sample_list)
        train_lines_out.append(l_new+'\n') if is_train else eval_lines_out.append(l_new+'\n')

    else:
        train_lines_out.append(l+'\n') if is_train else eval_lines_out.append(l+'\n')

f = open(transformed_train_log_file, 'w')
f.writelines(train_lines_out)
f.close()


f = open(transformed_eval_log_file, 'w')
f.writelines(eval_lines_out)
f.close()