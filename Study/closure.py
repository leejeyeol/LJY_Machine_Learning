def out_func():
    var = 1
    def inner_func():
        print(var)
    return inner_func

print('debug')