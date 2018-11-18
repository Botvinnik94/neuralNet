from random import randint

file = open("learningData.txt", "w")

for i in range(5001):
    k = randint(0,1)
    if k == 1:
        file.write("1,0\n")
    else:
        file.write("0,1\n")

file.close()


