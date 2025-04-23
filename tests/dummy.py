x = [False, None, True]

x = [False if el is None else el for el in x]

print(x)