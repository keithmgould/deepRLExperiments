import matplotlib.pyplot as plt
import pdb
import numpy as np

runs = []

runs.append([16, 21, 12, 15, 20, 20, 15, 32, 16, 21, 17, 13, 21, 39, 18, 18, 38, 16, 19, 12, 22, 36, 24, 17, 14, 38, 28, 17, 44, 19, 52, 21, 31, 50, 28, 76, 21, 29, 15, 36, 59, 29, 17, 74, 35, 55, 78, 21, 23, 10, 56, 14, 44, 65, 30, 47, 32, 46, 10, 37, 28, 66, 17, 50, 47, 20, 72, 33, 26, 16, 59, 20, 44, 32, 61, 90, 79, 20, 21, 48, 92, 84, 28, 98, 42, 36, 70, 55, 29, 46, 63, 48, 73, 20, 31, 35, 109, 132, 82, 115, 78, 48, 15, 84, 85, 92, 65, 90, 161, 102, 135, 99, 74, 66, 31, 100, 85, 192, 208, 177, 101, 74, 169, 46, 63, 117, 126, 79, 179, 48, 100, 36, 61, 24, 84, 77, 84, 72, 111, 112, 139, 88, 74, 82, 78, 130, 144, 53, 144, 81, 116, 58, 54, 83, 142, 167, 88, 166, 86, 120, 375, 80, 197, 91, 165, 96, 227, 169, 103, 373, 202, 98, 270, 193, 159, 374, 280, 123, 220, 65, 601, 464, 398, 269, 226, 284, 91, 516, 157, 414, 151, 221, 281, 309, 246, 516, 301, 233, 242, 119, 408, 91, 287, 27, 228, 251, 768, 138, 216, 147, 290, 581, 106, 195, 255, 281, 404, 461, 280, 299, 395, 135, 143, 401, 495, 274, 97, 401, 684, 622, 529, 128, 509, 360, 180, 557, 299, 626, 120, 182, 226, 367, 367, 98, 141, 159, 118, 598, 209, 271, 181, 308, 406, 228, 368, 170, 104, 285, 100, 672, 178, 112, 381, 840, 286, 218, 277, 119, 135, 517, 131, 235, 453, 330, 92, 231, 149, 550, 387, 419, 258, 115, 544, 628, 133, 278, 346, 176, 125, 229, 333, 220, 304, 154, 233, 119, 201, 459, 185, 211, 298, 331, 113, 115, 241, 135, 828, 429, 181, 599, 255, 111, 195, 269, 157, 779, 102, 123, 505, 152, 254, 176, 504, 284, 109, 718, 115, 527, 183, 181, 85, 275, 747, 670, 414, 131, 304, 120, 85, 431, 406, 460, 113, 224, 182, 520, 138, 350, 297, 322, 142, 220, 320, 140, 358, 117, 106, 316, 351, 119, 218, 211, 232, 236, 160, 128, 339, 115, 90, 161, 113, 145, 80, 177, 89, 152, 194, 125, 231, 202, 338, 274, 295, 257, 120, 112, 227, 321, 821, 88, 472, 216, 194, 345, 89, 267, 308, 282, 754, 322, 223, 636, 296, 322, 118, 267, 406, 326, 1000, 349, 506, 709, 119, 569, 396, 423, 715, 127, 600, 721, 230, 285, 405, 528, 354, 372, 321, 328, 823, 214, 350, 397, 1000, 462, 599, 743, 801, 416, 907, 293, 1000, 383, 322, 1000, 346, 658, 664, 756, 799, 389, 1000, 350, 1000, 852, 209, 596, 461, 637, 1000, 323, 1000, 481, 1000, 595, 704, 1000, 205, 499, 227, 430, 1000, 1000, 306, 594, 173, 1000, 1000, 1000, 1000, 886, 467, 1000, 792, 1000, 400, 1000, 1000, 804, 650, 1000, 795, 1000, 870, 527, 474, 1000, 1000, 1000, 805, 1000])

runs.append([24, 11, 17, 40, 52, 17, 13, 29, 46, 28, 28, 38, 12, 21, 17, 14, 20, 23, 70, 16, 22, 33, 21, 48, 12, 10, 35, 56, 30, 14, 33, 32, 13, 19, 26, 16, 41, 18, 37, 14, 14, 12, 76, 42, 10, 40, 14, 24, 47, 35, 80, 10, 25, 9, 15, 17, 21, 29, 20, 38, 12, 34, 18, 17, 17, 49, 21, 13, 41, 36, 29, 15, 10, 12, 31, 33, 51, 21, 18, 15, 28, 51, 13, 14, 56, 14, 25, 48, 17, 16, 17, 28, 46, 51, 47, 44, 22, 17, 42, 24, 11, 24, 16, 25, 20, 49, 51, 31, 32, 103, 14, 34, 26, 16, 49, 17, 45, 31, 28, 35, 26, 19, 81, 35, 18, 27, 17, 16, 10, 15, 28, 9, 63, 20, 34, 74, 21, 42, 50, 37, 24, 22, 19, 32, 20, 46, 24, 8, 61, 34, 11, 26, 54, 47, 117, 41, 21, 22, 59, 59, 62, 41, 22, 58, 49, 33, 24, 20, 23, 30, 43, 28, 17, 54, 93, 37, 92, 114, 56, 70, 81, 226, 34, 169, 27, 65, 39, 27, 37, 237, 80, 30, 65, 81, 160, 22, 85, 206, 22, 55, 108, 75, 57, 110, 221, 112, 154, 101, 194, 60, 98, 46, 98, 115, 303, 250, 60, 210, 106, 207, 134, 234, 164, 209, 81, 265, 320, 289, 96, 124, 212, 233, 176, 286, 245, 854, 451, 172, 179, 251, 285, 221, 235, 269, 551, 213, 527, 256, 430, 135, 464, 179, 358, 212, 412, 87, 296, 283, 474, 140, 109, 188, 185, 331, 109, 294, 311, 198, 232, 232, 107, 127, 185, 250, 285, 169, 164, 101, 169, 191, 132, 116, 295, 272, 366, 144, 136, 105, 381, 170, 90, 89, 107, 212, 113, 131, 218, 92, 41, 101, 133, 115, 426, 171, 108, 126, 134, 161, 132, 180, 132, 97, 368, 158, 164, 233, 104, 193, 306, 111, 328, 155, 208, 517, 246, 276, 341, 446, 254, 106, 479, 320, 239, 784, 475, 318, 253, 121, 181, 118, 676, 344, 381, 103, 206, 218, 344, 188, 286, 188, 313, 320, 432, 159, 191, 122, 300, 109, 359, 256, 329, 684, 203, 195, 511, 1000, 1000, 178, 1000, 307, 212, 128, 487, 528, 246, 769, 206, 520, 379, 260, 109, 194, 289, 189, 110, 333, 286, 365, 500, 682, 1000, 339, 833, 1000, 381, 638, 338, 567, 565, 638, 1000, 717, 272, 455, 1000, 536, 479, 1000, 1000, 779, 597, 1000, 1000, 1000, 436, 1000, 1000, 1000, 1000, 963, 677, 482, 1000, 1000, 763, 298, 625, 852, 1000, 344, 1000, 1000, 1000, 668, 681, 658, 874, 293, 672, 1000, 688, 758, 1000, 321, 634, 1000, 811, 473, 1000, 175, 1000, 1000, 948, 1000, 965, 1000, 1000, 1000, 1000, 1000, 639, 725, 913, 947, 245, 1000, 925, 1000, 1000, 1000, 1000, 1000, 378, 1000, 171, 503, 619, 513, 876, 538, 525, 328, 463, 740, 1000, 481, 474, 1000, 714, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 227, 1000, 101])

runs.append([11, 21, 13, 24, 26, 13, 10, 16, 33, 36, 21, 47, 15, 15, 12, 17, 17, 19, 12, 31, 21, 14, 13, 14, 28, 20, 17, 25, 13, 18, 12, 12, 51, 12, 32, 17, 29, 22, 40, 20, 64, 34, 22, 25, 26, 14, 16, 30, 22, 13, 46, 25, 24, 22, 21, 22, 11, 27, 34, 17, 19, 13, 31, 27, 43, 12, 13, 34, 21, 75, 45, 18, 57, 21, 16, 25, 26, 38, 42, 64, 45, 46, 51, 105, 46, 30, 39, 20, 13, 29, 16, 60, 49, 18, 27, 41, 25, 24, 40, 67, 8, 23, 17, 39, 24, 53, 51, 14, 31, 28, 32, 17, 18, 20, 88, 64, 30, 16, 24, 21, 28, 46, 14, 48, 51, 75, 73, 53, 30, 29, 36, 22, 31, 40, 18, 40, 34, 39, 58, 57, 39, 35, 34, 48, 93, 90, 49, 20, 29, 27, 39, 56, 34, 92, 56, 123, 45, 24, 61, 61, 90, 29, 26, 69, 43, 16, 72, 25, 57, 16, 119, 95, 48, 53, 85, 55, 143, 93, 88, 208, 113, 88, 231, 117, 180, 119, 194, 154, 85, 37, 114, 138, 87, 81, 261, 120, 252, 103, 104, 278, 198, 153, 43, 215, 85, 185, 192, 146, 115, 76, 274, 99, 127, 97, 81, 184, 178, 166, 384, 363, 220, 156, 162, 79, 257, 257, 132, 49, 90, 545, 213, 195, 172, 424, 287, 358, 420, 259, 485, 177, 171, 100, 448, 114, 164, 315, 302, 230, 292, 218, 239, 289, 219, 212, 121, 400, 230, 177, 296, 196, 379, 241, 120, 239, 163, 339, 182, 292, 332, 227, 155, 272, 237, 236, 186, 269, 134, 206, 126, 183, 362, 296, 383, 392, 389, 200, 537, 122, 320, 307, 193, 275, 165, 321, 420, 223, 86, 332, 253, 546, 387, 418, 94, 125, 156, 174, 145, 417, 124, 119, 153, 366, 336, 269, 228, 94, 209, 213, 135, 307, 620, 100, 92, 181, 91, 84, 193, 172, 223, 105, 167, 399, 91, 222, 121, 105, 91, 150, 181, 137, 265, 84, 323, 156, 103, 258, 354, 370, 126, 188, 185, 183, 349, 282, 220, 287, 209, 176, 328, 106, 526, 90, 322, 273, 449, 104, 319, 420, 303, 189, 195, 326, 193, 153, 177, 653, 436, 341, 332, 233, 334, 544, 323, 331, 535, 115, 120, 100, 121, 91, 222, 191, 212, 285, 200, 171, 203, 177, 92, 271, 145, 136, 113, 158, 161, 108, 126, 94, 162, 89, 93, 97, 96, 95, 80, 151, 181, 129, 156, 106, 96, 89, 97, 83, 141, 83, 109, 114, 120, 92, 89, 309, 202, 75, 123, 163, 83, 199, 109, 587, 79, 471, 308, 310, 766, 398, 162, 157, 200, 474, 228, 103, 213, 512, 349, 334, 193, 349, 214, 571, 249, 277, 297, 428, 304, 841, 494, 211, 99, 549, 92, 301, 324, 777, 295, 115, 155, 211, 591, 169, 643, 107, 177, 279, 811, 363, 307, 127, 590, 675, 504, 187, 562, 452, 386, 216, 105, 191, 371, 311])

runs.append([27, 11, 14, 30, 32, 16, 13, 13, 32, 15, 16, 9, 12, 26, 19, 23, 13, 26, 14, 30, 30, 10, 19, 8, 17, 16, 12, 29, 23, 31, 24, 19, 13, 26, 27, 41, 14, 11, 39, 19, 27, 30, 14, 18, 16, 11, 33, 22, 17, 13, 88, 12, 19, 23, 25, 36, 29, 14, 22, 36, 15, 20, 12, 18, 13, 31, 23, 26, 33, 51, 18, 32, 67, 21, 11, 19, 32, 37, 34, 51, 79, 38, 71, 28, 16, 17, 27, 39, 12, 88, 44, 70, 41, 47, 30, 25, 68, 79, 39, 52, 18, 42, 41, 29, 62, 120, 48, 44, 122, 43, 44, 62, 54, 54, 45, 178, 74, 56, 118, 68, 51, 36, 85, 58, 143, 90, 113, 146, 33, 81, 87, 70, 58, 70, 121, 107, 244, 83, 108, 126, 103, 105, 180, 119, 152, 189, 158, 326, 225, 84, 86, 204, 86, 115, 111, 189, 165, 95, 396, 169, 108, 35, 239, 137, 148, 96, 81, 163, 227, 89, 139, 82, 95, 141, 133, 77, 121, 99, 84, 192, 215, 98, 54, 156, 86, 158, 149, 139, 94, 254, 123, 289, 114, 225, 170, 310, 172, 550, 77, 165, 227, 196, 113, 511, 227, 248, 227, 275, 196, 402, 75, 130, 247, 224, 264, 143, 87, 136, 156, 284, 145, 95, 196, 71, 197, 145, 244, 360, 191, 564, 40, 117, 168, 410, 379, 280, 500, 731, 541, 97, 280, 259, 294, 453, 156, 414, 396, 396, 873, 175, 366, 83, 190, 276, 745, 371, 161, 136, 266, 121, 460, 227, 117, 226, 300, 146, 254, 179, 170, 105, 357, 171, 220, 118, 101, 307, 256, 102, 235, 113, 199, 191, 157, 175, 129, 359, 264, 129, 144, 300, 214, 649, 28, 131, 177, 453, 372, 149, 840, 1000, 464, 837, 410, 287, 93, 726, 299, 609, 297, 577, 238, 423, 222, 383, 277, 295, 455, 474, 251, 628, 232, 133, 241, 236, 107, 257, 686, 226, 481, 115, 668, 724, 431, 446, 271, 307, 1000, 1000, 379, 307, 133, 1000, 299, 525, 914, 683, 332, 563, 588, 395, 906, 503, 481, 1000, 275, 1000, 142, 1000, 585, 452, 879, 1000, 676, 259, 747, 225, 1000, 479, 226, 337, 883, 644, 585, 466, 1000, 1000, 1000, 803, 409, 1000, 177, 518, 1000, 1000, 693, 724, 1000, 248, 1000, 1000, 1000, 1000, 698, 663, 1000, 1000, 491, 826, 677, 349, 1000, 677, 434, 1000, 1000, 509, 1000, 280, 1000, 412, 650, 334, 736, 1000, 954, 1000, 1000, 1000, 608, 734, 702, 1000, 1000, 773, 207, 1000, 1000, 192, 634, 471, 602, 510, 965, 605, 222, 746, 749, 1000, 50, 352, 1000, 1000, 405, 1000, 735, 1000, 1000, 476, 1000, 349, 1000, 1000, 1000, 124, 1000, 740, 1000, 1000, 1000, 727, 825, 916, 1000, 192, 1000, 1000, 1000, 1000, 1000, 640, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 615, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 480, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 189, 1000, 1000, 1000])

runs.append([25, 25, 15, 16, 9, 57, 15, 16, 12, 21, 20, 12, 15, 18, 14, 14, 63, 12, 26, 12, 11, 24, 33, 34, 25, 20, 18, 28, 24, 34, 90, 26, 22, 19, 42, 17, 17, 14, 22, 65, 16, 20, 17, 23, 41, 40, 14, 27, 35, 39, 41, 17, 21, 26, 10, 35, 15, 27, 20, 58, 21, 23, 26, 19, 43, 20, 32, 57, 30, 95, 95, 26, 23, 41, 32, 34, 166, 92, 24, 41, 51, 42, 12, 56, 22, 57, 59, 24, 39, 31, 52, 36, 33, 15, 33, 32, 17, 19, 46, 33, 28, 32, 39, 122, 45, 37, 18, 23, 81, 19, 54, 64, 63, 132, 30, 86, 53, 42, 44, 108, 39, 90, 26, 54, 145, 42, 28, 99, 103, 35, 76, 113, 43, 98, 82, 21, 56, 109, 86, 165, 149, 28, 57, 37, 100, 64, 156, 64, 109, 60, 21, 40, 57, 171, 85, 48, 87, 68, 111, 78, 90, 82, 37, 140, 85, 45, 106, 22, 75, 95, 87, 105, 204, 93, 105, 159, 81, 96, 137, 113, 62, 32, 112, 98, 93, 112, 200, 125, 155, 86, 200, 98, 108, 208, 92, 80, 76, 83, 145, 225, 106, 134, 86, 113, 110, 126, 214, 113, 101, 176, 99, 141, 300, 124, 186, 206, 238, 106, 193, 100, 185, 38, 109, 77, 272, 219, 276, 289, 232, 92, 339, 137, 207, 109, 241, 126, 102, 264, 250, 196, 300, 85, 130, 87, 76, 137, 247, 169, 106, 81, 176, 121, 95, 98, 174, 328, 282, 192, 473, 487, 167, 580, 871, 123, 425, 201, 294, 192, 104, 156, 131, 241, 104, 109, 106, 120, 100, 101, 298, 317, 260, 216, 92, 157, 291, 334, 235, 186, 176, 152, 90, 264, 268, 261, 91, 125, 266, 118, 294, 385, 171, 180, 209, 90, 294, 326, 193, 218, 306, 237, 208, 158, 340, 423, 428, 321, 254, 188, 306, 112, 374, 322, 151, 408, 453, 344, 355, 108, 528, 220, 679, 120, 586, 274, 550, 704, 498, 441, 711, 392, 409, 191, 671, 777, 492, 917, 416, 429, 137, 462, 338, 412, 442, 211, 239, 502, 188, 180, 1000, 490, 370, 268, 549, 108, 691, 518, 336, 284, 332, 1000, 1000, 426, 556, 404, 739, 762, 682, 846, 600, 496, 368, 751, 209, 496, 285, 1000, 433, 469, 193, 986, 203, 1000, 379, 254, 752, 653, 989, 216, 631, 453, 526, 444, 326, 181, 552, 252, 120, 406, 240, 335, 239, 280, 201, 266, 254, 395, 295, 356, 701, 510, 192, 805, 571, 525, 335, 388, 191, 1000, 1000, 436, 566, 1000, 854, 486, 483, 950, 592, 274, 258, 631, 1000, 292, 732, 210, 476, 970, 488, 284, 733, 798, 210, 803, 305, 1000, 405, 1000, 1000, 397, 775, 840, 1000, 275, 595, 1000, 1000, 444, 666, 1000, 391, 471, 196, 359, 1000, 1000, 575, 608, 1000, 736, 264, 1000, 1000, 1000, 1000, 1000, 1000, 527, 738, 955, 602, 1000, 1000, 474, 1000, 1000, 674, 897, 806, 847, 690, 725])

for x in range(len(runs)):
  subject = runs[x]
  poop = []
  for i in range(len(subject)):
    sub_sample = subject[0:i][-20:]
    poop.append(np.mean(sub_sample))
  plt.plot(poop, '.')
plt.savefig('baseline_results_20.png')
