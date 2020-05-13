class option:
    def __init__(self, N, L, C, scale, seed, nCV=0, ratio=0,mode='merged', drop=0):
        self.N = N
        self.L = L
        self.C = C
        self.scale = scale
        self.seed = seed
        self.nCV = nCV
        self.drop =drop
        self.ratio = ratio
        self.mode = mode


