# TODO: Add     & 0xff      mask to assignments with a byte typed variable as target.

# print(534 & 0xff)
# print(-534 & 0xff)
# print(256 & 0xff)
#
#
# print(((289 % 987) - ((-2876)*81211)) & 0xff)
# print(((((289 & 0xff) % (987 & 0xff) & 0xff) - ((((-2876) & 0xff) * (81211 & 0xff)) & 0xff)) & 0xff))
#
# print(534 + 1 < 780)
# print((534 + 1) & 0xff < 780 & 0xff)
#
#
# print("a" or None or None)

print("" or "")
print("-" or "")


print(["x", "x", "y"] <= ["x", "x"])
print(["x", "x"] <= ["x", "x", "y"])
print(["x", "x"] <= ["x", "y"])

from collections import Counter


def issubset(X, Y):
    return len(Counter(X)-Counter(Y)) == 0


print(issubset(["x", "x"], ["x", "y"]))
print(issubset(["x", "y"], ["x", "x", "y"]))
print(issubset([], ["x", "x", "y"]))

