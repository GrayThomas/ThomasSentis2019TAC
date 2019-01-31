def flipadd(ori):
    N = len(ori)
    for i in reversed(range(N)):
        ori.append((ori[i][0],ori[i][1],-ori[i][2]))
    # ori.append((ori[-1][0],0*ori[-1][1],0*ori[-1][2]))
def append_zero(ori):
    ori.append((ori[-1][0],0*ori[-1][1],0*ori[-1][2]))

def append_duplicate(ori):
    ori.append((ori[-1][0],ori[-1][1],ori[-1][2]))

def prepend_zero(ori):
    ori.reverse()
    append_zero(ori)
    ori.reverse()

def prepend_duplicate(ori):
    ori.reverse()
    append_duplicate(ori)
    ori.reverse()

def close_shape_reflect_imaginary(ori,orid):
    prepend_duplicate(ori)
    prepend_zero(orid)
    append_duplicate(ori)
    append_zero(orid)
    flipadd(ori)
    flipadd(orid)

def ori4_flipadd(ori4):
    N = len(ori4)
    for i in reversed(range(N)):
        ori4.append((ori4[i][0], ori4[i][1],-ori4[i][2], ori4[i][3]))
    # ori.append((ori[-1][0],0*ori[-1][1],0*ori[-1][2]))
    return ori4

def ori4_cap(ori4):
    ori4.append((ori4[-1][0],ori4[-1][1],ori4[-1][2],0.0))
    return ori4

def ori4_close(ori4):
    ori4_cap(ori4)
    ori4.reverse()
    ori4_cap(ori4)
    ori4.reverse()
    return ori4

def close_shape_reflect_imaginary_ori4(ori4):
    ori4_cap(ori4)
    ori4.reverse()
    ori4_cap(ori4)
    ori4.reverse()
    ori4_flipadd(ori4)
    return ori4