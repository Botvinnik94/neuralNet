from random import randint

file = open("learningData.txt", "w")

for i in range(20000):
    k = randint(0,3)
    if k == 0:
        file.write("1,0,0,1\n")
    elif k == 1:
        file.write("0,1,0,1\n")
    elif k == 2:
        file.write("1,1,1,0\n")
    else:
        file.write("0,0,1,0\n")

file.close()


