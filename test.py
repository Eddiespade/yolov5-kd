import torch
import torch.nn as nn

'''
tensor([[ 0.00000,  2.00000,  0.49078,  0.09485,  0.39443,  0.18970],
        [ 0.00000, 14.00000,  0.33598,  0.79206,  0.67196,  0.41587],
        [ 0.00000, 14.00000,  0.32796,  0.80649,  0.65592,  0.38701],
        [ 0.00000, 14.00000,  0.68205,  0.84337,  0.63589,  0.31325],
        [ 1.00000,  4.00000,  0.16341,  0.45297,  0.02617,  0.07705],
        [ 1.00000,  4.00000,  0.18304,  0.45152,  0.03053,  0.07995],
        [ 1.00000,  4.00000,  0.20339,  0.45224,  0.03343,  0.07850],
        [ 1.00000,  4.00000,  0.22592,  0.45079,  0.02907,  0.07850],
        [ 1.00000,  4.00000,  0.24627,  0.45152,  0.02617,  0.07705],
        [ 1.00000,  4.00000,  0.26953,  0.45079,  0.02907,  0.07850],
        [ 1.00000,  4.00000,  0.28988,  0.44934,  0.02617,  0.07850],
        [ 1.00000,  4.00000,  0.31023,  0.44788,  0.02617,  0.07850],
        [ 1.00000,  4.00000,  0.33131,  0.45006,  0.02762,  0.07705],
        [ 1.00000,  4.00000,  0.35312,  0.44788,  0.02762,  0.07850],
        [ 1.00000,  4.00000,  0.37492,  0.44570,  0.02762,  0.07995],
        [ 1.00000,  4.00000,  0.39745,  0.44497,  0.02617,  0.07850],
        [ 1.00000,  4.00000,  0.41853,  0.44570,  0.02471,  0.07705],
        [ 1.00000,  4.00000,  0.43888,  0.44497,  0.02762,  0.07850],
        [ 1.00000,  4.00000,  0.45996,  0.44425,  0.02326,  0.07995],
        [ 1.00000,  4.00000,  0.48104,  0.44352,  0.02471,  0.07850],
        [ 1.00000,  4.00000,  0.50357,  0.44425,  0.02617,  0.07995],
        [ 1.00000,  4.00000,  0.61623,  0.15424,  0.02471,  0.06396],
        [ 1.00000, 14.00000,  0.56317,  0.36938,  0.47681,  0.43611],
        [ 1.00000, 14.00000,  0.80303,  0.43407,  0.14537,  0.30673],
        [ 1.00000, 11.00000,  0.52029,  0.85953,  0.15554,  0.10593],
        [ 2.00000,  6.00000,  0.96875,  0.48110,  0.06251,  0.53131],
        [ 2.00000,  4.00000,  0.49145,  0.54184,  0.02925,  0.07495],
        [ 2.00000,  8.00000,  0.78759,  0.45410,  0.25958,  0.22485],
        [ 2.00000,  8.00000,  0.40462,  0.43216,  0.25044,  0.22485],
        [ 2.00000, 17.00000,  0.22548,  0.56926,  0.40399,  0.50636],
        [ 3.00000,  1.00000,  0.11535,  0.06756,  0.23069,  0.13513],
        [ 3.00000, 14.00000,  0.61691,  0.06756,  0.45652,  0.13513],
        [ 3.00000, 14.00000,  0.90262,  0.05021,  0.19475,  0.10041],
        [ 3.00000, 10.00000,  0.09799,  0.61074,  0.19598,  0.35064],
        [ 3.00000, 14.00000,  0.07369,  0.40592,  0.14737,  0.26732],
        [ 3.00000, 14.00000,  0.18738,  0.53697,  0.37477,  0.49818],
        [ 3.00000, 14.00000,  0.13435,  0.45278,  0.21698,  0.30550],
        [ 3.00000, 13.00000,  0.71168,  0.46667,  0.57663,  0.43743],
        [ 3.00000, 14.00000,  0.42424,  0.37294,  0.09547,  0.12845],
        [ 4.00000, 13.00000,  0.70793,  0.13500,  0.37930,  0.27001],
        [ 4.00000, 14.00000,  0.64420,  0.12503,  0.37469,  0.25005],
        [ 4.00000, 17.00000,  0.66854,  0.72987,  0.66292,  0.35780],
        [ 4.00000, 14.00000,  0.85051,  0.70837,  0.29898,  0.41001],
        [ 4.00000, 14.00000,  0.62040,  0.71681,  0.34552,  0.43919],
        [ 4.00000, 14.00000,  0.44457,  0.71681,  0.21499,  0.44840],
        [ 4.00000,  1.00000,  0.13552,  0.74515,  0.27105,  0.27272],
        [ 4.00000, 14.00000,  0.06105,  0.68080,  0.12209,  0.40142],
        [ 5.00000,  6.00000,  0.89579,  0.27837,  0.20842,  0.47696],
        [ 5.00000,  6.00000,  0.70763,  0.14142,  0.23886,  0.20778],
        [ 5.00000,  2.00000,  0.22197,  0.21196,  0.44394,  0.39704],
        [ 5.00000, 14.00000,  0.16285,  0.83763,  0.32569,  0.32473],
        [ 6.00000, 14.00000,  0.61508,  0.38554,  0.03319,  0.08581],
        [ 6.00000,  8.00000,  0.48869,  0.40423,  0.02538,  0.04066],
        [ 6.00000,  8.00000,  0.53504,  0.40566,  0.02745,  0.04558],
        [ 6.00000,  8.00000,  0.52727,  0.41355,  0.03263,  0.02978],
        [ 6.00000, 18.00000,  0.59941,  0.51522,  0.06452,  0.03940],
        [ 7.00000,  6.00000,  0.20787,  0.53494,  0.04712,  0.03502],
        [ 7.00000,  6.00000,  0.07134,  0.54279,  0.06645,  0.04830],
        [ 7.00000, 14.00000,  0.57092,  0.53434,  0.22230,  0.39364],
        [ 7.00000, 14.00000,  0.79262,  0.56875,  0.14860,  0.32481],
        [ 7.00000, 14.00000,  0.70684,  0.49992,  0.07370,  0.15576],
        [ 7.00000, 14.00000,  0.68026,  0.60558,  0.11478,  0.12799],
        [ 7.00000, 14.00000,  0.41265,  0.56694,  0.08457,  0.32843],
        [ 7.00000,  2.00000,  0.11692,  0.92052,  0.23384,  0.15896],
        [ 7.00000, 14.00000,  0.49178,  0.90726,  0.24284,  0.18548],
        [ 7.00000, 14.00000,  0.61743,  0.91934,  0.32499,  0.16132],
        [ 7.00000, 14.00000,  0.83853,  0.90303,  0.26942,  0.19394],
        [ 7.00000,  4.00000,  0.82765,  0.89123,  0.02054,  0.03987],
        [ 7.00000,  4.00000,  0.84940,  0.88338,  0.02295,  0.05799]], device='cuda:0')
'''

a = torch.tensor([[0.00000, 2.00000, 0.49078, 0.09485, 0.39443, 0.18970],
                  [0.00000, 14.00000, 0.33598, 0.79206, 0.67196, 0.41587],
                  [0.00000, 14.00000, 0.32796, 0.80649, 0.65592, 0.38701],
                  [0.00000, 14.00000, 0.68205, 0.84337, 0.63589, 0.31325],
                  [1.00000, 4.00000, 0.16341, 0.45297, 0.02617, 0.07705],
                  [1.00000, 4.00000, 0.18304, 0.45152, 0.03053, 0.07995],
                  [1.00000, 4.00000, 0.20339, 0.45224, 0.03343, 0.07850],
                  [1.00000, 4.00000, 0.22592, 0.45079, 0.02907, 0.07850],
                  [1.00000, 4.00000, 0.24627, 0.45152, 0.02617, 0.07705],
                  [1.00000, 4.00000, 0.26953, 0.45079, 0.02907, 0.07850],
                  [1.00000, 4.00000, 0.28988, 0.44934, 0.02617, 0.07850],
                  [1.00000, 4.00000, 0.31023, 0.44788, 0.02617, 0.07850],
                  [1.00000, 4.00000, 0.33131, 0.45006, 0.02762, 0.07705],
                  [1.00000, 4.00000, 0.35312, 0.44788, 0.02762, 0.07850],
                  [1.00000, 4.00000, 0.37492, 0.44570, 0.02762, 0.07995],
                  [1.00000, 4.00000, 0.39745, 0.44497, 0.02617, 0.07850],
                  [1.00000, 4.00000, 0.41853, 0.44570, 0.02471, 0.07705],
                  [1.00000, 4.00000, 0.43888, 0.44497, 0.02762, 0.07850],
                  [1.00000, 4.00000, 0.45996, 0.44425, 0.02326, 0.07995],
                  [1.00000, 4.00000, 0.48104, 0.44352, 0.02471, 0.07850],
                  [1.00000, 4.00000, 0.50357, 0.44425, 0.02617, 0.07995],
                  [1.00000, 4.00000, 0.61623, 0.15424, 0.02471, 0.06396],
                  [1.00000, 14.00000, 0.56317, 0.36938, 0.47681, 0.43611],
                  [1.00000, 14.00000, 0.80303, 0.43407, 0.14537, 0.30673],
                  [1.00000, 11.00000, 0.52029, 0.85953, 0.15554, 0.10593],
                  [2.00000, 6.00000, 0.96875, 0.48110, 0.06251, 0.53131],
                  [2.00000, 4.00000, 0.49145, 0.54184, 0.02925, 0.07495],
                  [2.00000, 8.00000, 0.78759, 0.45410, 0.25958, 0.22485],
                  [2.00000, 8.00000, 0.40462, 0.43216, 0.25044, 0.22485],
                  [2.00000, 17.00000, 0.22548, 0.56926, 0.40399, 0.50636],
                  [3.00000, 1.00000, 0.11535, 0.06756, 0.23069, 0.13513],
                  [3.00000, 14.00000, 0.61691, 0.06756, 0.45652, 0.13513],
                  [3.00000, 14.00000, 0.90262, 0.05021, 0.19475, 0.10041],
                  [3.00000, 10.00000, 0.09799, 0.61074, 0.19598, 0.35064],
                  [3.00000, 14.00000, 0.07369, 0.40592, 0.14737, 0.26732],
                  [3.00000, 14.00000, 0.18738, 0.53697, 0.37477, 0.49818],
                  [3.00000, 14.00000, 0.13435, 0.45278, 0.21698, 0.30550],
                  [3.00000, 13.00000, 0.71168, 0.46667, 0.57663, 0.43743],
                  [3.00000, 14.00000, 0.42424, 0.37294, 0.09547, 0.12845],
                  [4.00000, 13.00000, 0.70793, 0.13500, 0.37930, 0.27001],
                  [4.00000, 14.00000, 0.64420, 0.12503, 0.37469, 0.25005],
                  [4.00000, 17.00000, 0.66854, 0.72987, 0.66292, 0.35780],
                  [4.00000, 14.00000, 0.85051, 0.70837, 0.29898, 0.41001],
                  [4.00000, 14.00000, 0.62040, 0.71681, 0.34552, 0.43919],
                  [4.00000, 14.00000, 0.44457, 0.71681, 0.21499, 0.44840],
                  [4.00000, 1.00000, 0.13552, 0.74515, 0.27105, 0.27272],
                  [4.00000, 14.00000, 0.06105, 0.68080, 0.12209, 0.40142],
                  [5.00000, 6.00000, 0.89579, 0.27837, 0.20842, 0.47696],
                  [5.00000, 6.00000, 0.70763, 0.14142, 0.23886, 0.20778],
                  [5.00000, 2.00000, 0.22197, 0.21196, 0.44394, 0.39704],
                  [5.00000, 14.00000, 0.16285, 0.83763, 0.32569, 0.32473],
                  [6.00000, 14.00000, 0.61508, 0.38554, 0.03319, 0.08581],
                  [6.00000, 8.00000, 0.48869, 0.40423, 0.02538, 0.04066],
                  [6.00000, 8.00000, 0.53504, 0.40566, 0.02745, 0.04558],
                  [6.00000, 8.00000, 0.52727, 0.41355, 0.03263, 0.02978],
                  [6.00000, 18.00000, 0.59941, 0.51522, 0.06452, 0.03940],
                  [7.00000, 6.00000, 0.20787, 0.53494, 0.04712, 0.03502],
                  [7.00000, 6.00000, 0.07134, 0.54279, 0.06645, 0.04830],
                  [7.00000, 14.00000, 0.57092, 0.53434, 0.22230, 0.39364],
                  [7.00000, 14.00000, 0.79262, 0.56875, 0.14860, 0.32481],
                  [7.00000, 14.00000, 0.70684, 0.49992, 0.07370, 0.15576],
                  [7.00000, 14.00000, 0.68026, 0.60558, 0.11478, 0.12799],
                  [7.00000, 14.00000, 0.41265, 0.56694, 0.08457, 0.32843],
                  [7.00000, 2.00000, 0.11692, 0.92052, 0.23384, 0.15896],
                  [7.00000, 14.00000, 0.49178, 0.90726, 0.24284, 0.18548],
                  [7.00000, 14.00000, 0.61743, 0.91934, 0.32499, 0.16132],
                  [7.00000, 14.00000, 0.83853, 0.90303, 0.26942, 0.19394],
                  [7.00000, 4.00000, 0.82765, 0.89123, 0.02054, 0.03987],
                  [7.00000, 4.00000, 0.84940, 0.88338, 0.02295, 0.05799]], device="cuda:0")


def creat_mask(cur, labels):
    B, H, W = cur.size()
    x, y, w, h = labels[2:]
    x1 = int(((x - w / 2) * W).ceil().cpu().numpy())
    x2 = int(((x + w / 2) * W).floor().cpu().numpy())
    y1 = int(((y - h / 2) * W).ceil().cpu().numpy())
    y2 = int(((y + h / 2) * W).floor().cpu().numpy())
    cur[labels[0].cpu().numpy()][y1: y2, x1: x2] = 1


# 根据a构建掩码矩阵
cur_mask = torch.full((8, 30, 30), 0.3, device="cuda:0")
for label in a:
    creat_mask(cur_mask, label)
    c = cur_mask[0]

print(cur_mask)