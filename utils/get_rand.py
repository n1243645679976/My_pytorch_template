import torch
def get_random(conf):
    type, conf = conf.split('#')
    if type == 'uniform_random':
        def gen_random(conf):
            min, max = conf.split('~')
            min, max = float(min), float(max)
            def random():
                return torch.rand(1) * (max - min) + min
            return random
    
    if type == 'gaussian_random':
        def gen_random(conf):
            mean, var = conf.split(',')
            mean, var = float(mean), float(var)
            def random():
                return torch.randn(1) * var + mean
            return random
    return gen_random(conf)

if __name__ == '__main__':
    a = get_random('uniform_random#1~2')
    b = get_random('uniform_random#3~4')
    print(a())
    print(a())
    print(b())
    print(a())

    a = get_random('gaussian_random#1,0.1')
    b = get_random('gaussian_random#3,0.1')
    print(a())
    print(a())
    print(b())
    print(a())

