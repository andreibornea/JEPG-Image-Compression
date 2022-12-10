import math


def ppm_tokenize(file):
    ok = False
    for character in file:
        character.strip()
        character = character.split('#')
        for i in character:
            if '"' not in i:
                s = ''
                for j in i:
                    if j.strip().isdigit() == True:
                        s = s+j.strip()
                    elif j == ' ' and s.strip() >= '0':
                        if ok == False:
                            yield 'P'+s.strip()
                        else:
                            yield s.strip()
                        ok = True
                        s = ''


def ppm_load(file):
    s = []
    for i in ppm_tokenize(file):
        s.append(i)
    w = int(s[1])
    h = int(s[2])
    img = []
    for j in range(h):
        trip = []
        for i in range(w):
            trip.append(
                (int(s[3+j*w + i*3]), int(s[j*w+i*3 + 4]), int(s[j*w+i*3 + 5])))
        img.append(trip)
    return (w, h, img)


def ppm_save(w, h, img, output):
    with open(output, 'w') as file:
        file.write('P3'+'\n')
        file.write(str(w)+'\n')
        file.write(str(h)+'\n')
        file.write('255'+'\n')
        for (i, j, k) in img:
            file.write(str(i)+' '+str(j)+' '+str(k)+' '+'\n')


def RGB2YCbCr(r, g, b):
    Y = 0 + 0.299 * r + 0.587 * g + 0.114 * b
    Cb = 128 - 0.168736 * r - 0.331264*g + 0.5*b
    Cr = 128 + 0.5 * r - 0.418688*g - 0.081312*b
    Y = round(Y)
    Cb = round(Cb)
    Cr = round(Cr)
    Y = max(0, min(255, Y))
    Cb = max(0, min(255, Cb))
    Cr = max(0, min(255, Cr))

    return (Y, Cb, Cr)


def YCbCr2RGB(Y, Cb, Cr):
    r = Y + 1.402*(Cr-128)
    g = Y - 0.344136 * (Cb-128) - 0.714136*(Cr-128)
    b = Y + 1.772 * (Cb-128)
    r = round(r)
    g = round(g)
    b = round(b)
    r = max(0, min(255, r))
    g = max(0, min(255, g))
    b = max(0, min(255, b))

    return (r, g, b)


def img_RGB2YCbCr(img):
    (w, h, trip) = ppm_load(img)
    Y = [[0]*w]*h
    Cb = [[0]*w]*h
    Cr = [[0]*w]*h
    for i in range(w):
        for j in range(h):
            (a, b, c) = trip[i*(w-1)+j]
            (Y[i][j], Cb[i][j], Cr[i][j]) = RGB2YCbCr(a, b, c)
    return(Y, Cb, Cr)


def img_YCbCr2RGB(Y, Cb, Cr):
    n = len(Y)
    for i in Y:
        m = len(i)
    s = []
    for i in range(n):
        for j in range(m):
            s.append(YCbCr2RGB(Y[i][j], Cb[i][j], Cr[i][j]))
    return (n, m, s)


def subsampling(w, h, C, a, b):
    m = []
    for i in range(h):
        t = 0
        s = 0
        l = []
        while t < w:
            if t % a == a-1:
                s = s+C[i][t]
                s = s/a
                l.append(s)
                s = 0
            else:
                s = s+C[i][t]
            t += 1
        s = s/(w % a)
        l.append(s)
        m.append(l)
    l = [[] for _ in range(h//b+1)]
    for i in range(len(m[1])):
        t = 0
        s = 0
        while t < h:
            if t % b == b-1:
                s = s+m[t][i]
                s = s/b
                l[t//b].append(s)
                s = 0
            else:
                s = s+m[t][i]
            t += 1
        s = s/(h % b)
        l[h//b].append(s)

    for i in range(w//a+1):
        for j in range(h//b+1):
            l[i][j] = round(l[i][j])
    return l


def extrapolate(w, h, C, a, b):
    m = [[] for _ in range(h//b+1)]
    if w % a == 0:
        for i in range(len(C)):
            for j in range(len(C[i])):
                for k in range(a):
                    m[i].append(C[i][j])
    else:
        for i in range(len(C)):
            for j in range(len(C[i])-1):
                for k in range(a):
                    m[i].append(C[i][j])
            for j in range(w % a):
                m[i].append(C[i][len(C[i])-1])
    ext = []
    if h % b == 0:
        for i in range(h//b):
            for j in range(b):
                ext.append(m[i])
    else:
        for i in range(h//b):
            for j in range(b):
                ext.append(m[i])
        for j in range(h % b):
            ext.append(m[h//b])
    return ext


def block_splitting(w, h, C):
    s = []
    for i in C:
        s.append(i) 
    if w%8!=0:
        for i in s:
            for j in range(8-w % 8):
                i.append(i[w-1])
    if h%8!=0:
        for j in range(8-h % 8):
            s.append(s[h-1])
    w=len(s[1])
    h=len(s)
    lm = [[[] for _ in range(8)] for __ in range((h//8)*(w//8))]
    for i in range(len(s)):
        for j in range(len(s[i])):
            lm[(w//8)*(i//8) + (j//8)][i % 8].append(s[i][j])
    for i in lm:
        yield i

def mat_aditional(n):
    coef = []
    for i in range(n):
        coef.append(math.pi * (i + 1/2)/n)
    mat = []
    mat.append([1/math.sqrt(n) for _ in range(n)])
    for i in range(1, n):
        l = []
        for j in range(n):
            l.append(math.cos(coef[j]*i) / math.sqrt(n)*math.sqrt(2))
        mat.append(l)
    return mat


def tran(mat):
    matt = []
    for i in mat:
        matt.append(i)
    for j in range(len(mat)):
        for i in range(j+1):
            k = matt[i][j]
            matt[i][j] = matt[j][i]
            matt[j][i] = k

    return matt


def product(v, mat):
    l = []
    for i in range(len(mat[0])):
        p = []
        for j in range(len(mat)):
            p.append(mat[j][i]*v[j])
        l.append(sum(p))
    return l


def DCT(v):
    C = mat_aditional(len(v))
    Ct = tran(C)
    return product(v, Ct)


def IDCT(v):
    C = mat_aditional(len(v))
    return product(v, C)


def mat_prod(A, B):
    l = []
    for i in A:
        l.append(product(i, B))
    return l


def DCT2(m, n, A):
    C = mat_aditional(m)
    c2 = mat_aditional(n)
    Ct = tran(c2)
    return mat_prod(mat_prod(C, A), Ct)


def IDCT2(m, n, A):
    C = mat_aditional(m)
    c2 = mat_aditional(n)
    Ct = tran(C)
    return mat_prod(mat_prod(Ct, A), c2)


def redalpha(i):
    k = i % 16
    c = i//16
    if k >= 8:
        k = 8-k % 8
        c += 1
    if c % 2 == 1:
        return(-1, k)
    else:
        return(1, k)


def ncoeff8(i, j):
    if i == 0:
        return (1, 4)
    else:
        return redalpha(i*(2*j+1))


def M8_to_str(M8):
    def for1(s, i):
        return f"{'+' if s >= 0 else '-'}{i:d}"

    return "\n".join(
        " ".join(for1(s, i) for (s, i) in row)
        for row in M8
    )

    print(M8_to_str(M8))


def alfak():
    a = []
    for i in range(8):
        a.append(math.cos((math.pi*i)/16))
    return a


def mat_aux():
    a = alfak()
    M = [[] for i in range(8)]
    M[0] = [1/math.sqrt(2) for _ in range(8)]
    for i in range(1, 8):
        for j in range(8):
            (semn, k) = ncoeff8(i, j)
            M[i].append(a[k]*semn)
    return M


def DCT_Chen_vect(v):
    C = mat_aux()
    A = []
    s=0
    for i in v:
        s = s+ i
    s = s * C[0][0]
    A.append(s/2)
    for i in range(1, 8):
        s = 0
        if i % 4 == 2:
            s = (v[0]+v[7]-v[3]-v[4])*C[i][0]+(v[1]+v[6]-v[2]-v[5])*C[i][1]
        elif i == 4:
            s = (v[0]-v[1]-v[2]+v[3]+v[4]-v[5]-v[6]+v[7])*C[i][0]
        else:
            for j in range(4):
                s = s + (v[j]-v[7-j]) * C[i][j]
        A.append(s/2)
    return A


def DCT_Chen(A):
    M = []
    for i in range(8):
        M.append(DCT_Chen_vect(A[i]))
    Matc=[[0 for i in range(8)] for _ in range(8)]
    for i in range(8):
        l=[]
        for j in range(8):
            l.append(M[j][i])
        l= DCT_Chen_vect(l)
        for j in range(8):
            Matc[i][j]=l[j]
    return tran(Matc)


def mataux():
    a = alfak()
    M = []
    M.append([a[4], a[4], a[4], a[4], a[4], a[4], a[4], a[4]])
    M.append([a[2], a[6], a[6]*-1, a[2]*-1, a[2]*-1, a[6]*-1, a[6], a[2]])
    M.append([a[4], -1*a[4], -1*a[4], a[4], a[4], -1*a[4], -1*a[4], a[4]])
    M.append([a[6], -1*a[2], a[2], -1*a[6], a[6]*-1, a[2], a[2]*-1, a[6]])
    M.append([a[1], a[3], a[5], a[7], -1*a[7], -1*a[5], -1*a[3], -1*a[1]])
    M.append([a[3], -1*a[7], -1*a[1], -1*a[5], a[5], a[1], a[7], a[3]*-1])
    M.append([a[5], -1*a[1], a[7], a[3], -1*a[3], a[7]*-1, a[1], -1*a[5]])
    M.append([a[7], -1*a[5], a[3], -1*a[1], a[1], a[3]*-1, a[5], a[7]*-1])
    return M


def v_s(v):
    return [v[0], v[2], v[4], v[6], v[1], v[3], v[5], v[7]]


def IDCT_Chen_vect(v1):
    C = mataux()
    v = v_s(v1)
    A = [0 for _ in range(8)]
    for i in range(4):
        s1 = 0
        if i in [0, 3]:
            s1 = (v[0]+v[2])*C[0][i]+v[1]*C[1][i]+v[3]*C[3][i]
        elif i in [1, 2]:
            s1 = (v[0]-v[2])*C[0][i]+v[1]*C[1][i]+v[3]*C[3][i]
        s2 = 0
        for j in range(4, 8):
            s2 += C[j][i]*v[j]
        A[i] = (s1+s2)/2
        A[7-i] = (s1-s2)/2
    return A


def IDCT_Chen(A):
    M = []
    for i in range(8):
        M.append(IDCT_Chen_vect(A[i]))
    Matc=[[0 for i in range(8)] for _ in range(8)]
    for i in range(8):
        l=[]
        for j in range(8):
            l.append(M[j][i])
        l=IDCT_Chen_vect(l)
        for j in range(8):
            Matc[i][j]=l[j]
    return tran(Matc)


def quantization(A, Q):
    M = []
    for i in range(8):
        l = []
        for j in range(8):
            l.append(round(A[i][j]/Q[i][j]))
        M.append(l)
    return M


def quantizationI(A, Q):
    M = []
    for i in range(8):
        l = []
        for j in range(8):
            l.append(round(A[i][j]*Q[i][j]))
        M.append(l)
    return M


def lum_chan():
    return [[16, 11, 10, 16, 24, 40, 51, 61],
            [12, 12, 14, 19, 26, 58, 60, 55],
            [14, 13, 16, 24, 40, 57, 69, 56],
            [14, 17, 22, 29, 51, 87, 80, 62],
            [18, 22, 37, 56, 68, 109, 103, 77],
            [24, 35, 55, 64, 81, 104, 113, 92],
            [49, 64, 78, 87, 103, 121, 120, 101],
            [72, 92, 95, 98, 112, 100, 103, 99]]


def chrom_chan():
    M = [[99 for _ in range(8)] for _ in range(8)]
    M[0][0] = 17
    M[0][1] = 18
    M[0][2] = 24
    M[0][3] = 47
    M[1][0] = 18
    M[1][1] = 21
    M[1][2] = 26
    M[1][3] = 66
    M[2][0] = 24
    M[2][1] = 26
    M[2][2] = 56
    M[2][3] = 56
    M[3][0] = 47
    M[3][1] = 66
    return M


def lc(phi):
    if phi >= 50:
        return 200-2*phi
    else:
        return 5000//phi


def QL(Q, phi):
    M = []
    for i in range(8):
        l = []
        for j in range(8):
            l.append(math.ceil((50+lc(phi)*Q[i][j])/100))
        M.append(l)
    return M


def Qmatrix(isY, phi):
    if isY is True:
        M = lum_chan()
        return QL(M, phi)
    else:
        M = chrom_chan()
        return QL(M, phi)


def zigzag(A):
    for k in range(16):
        if k%2==1:
            for i in range(8):
                for j in range(8):
                    if i+j == k:
                        yield A[i][j]
        else:
            for i in range(7, -1, -1):
                for j in range(7, -1, -1):
                    if i+j == k:
                        yield A[i][j]
         
def rle0(g):
    l = []
    nr = 0
    for i in g:
        if i == 0:
            nr += 1
        else:
            l.append((nr, i))
            nr = 0
    return l


        