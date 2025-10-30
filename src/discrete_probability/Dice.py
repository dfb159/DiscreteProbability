from discrete_probability import Probs, uniform

def D(n: int) -> Probs:
    return uniform(n) + 1

D4 = D(4)
D6 = D(6)
D8 = D(8)
D10 = D(10)
D12 = D(12)
D20 = D(20)
D100 = D(100)
