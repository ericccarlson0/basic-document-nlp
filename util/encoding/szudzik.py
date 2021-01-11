import math

def pair_ints(x, y):
    if x > y:
        return x * x + x + y
    else:
        return x + y * y

def unpair_int(z):
    floor_root_z = math.floor(math.sqrt(z))

    if z - (floor_root_z ** 2) < floor_root_z:
        x = z - (floor_root_z ** 2)
        y = floor_root_z
    else:
        x = floor_root_z
        y = z - (floor_root_z ** 2) - floor_root_z

    return x, y