# Comportamentos determinísticos

def p(a,b,x,y,lamb):

    if (lamb[x] == a) and (lamb[y + 2] == b):
        return 1
    
    else:
        return 0
    
# Correlatores
    
def E(x,y,lamb):

    soma = 0
    for a in [-1,1]:
        for b in [-1,1]:
            soma += a*b*p(a,b,x,y,lamb)
    
    return soma

def E_x(x,lamb):

    soma = 0
    for a in [-1,1]:
        for b in [-1,1]:
            soma += a*p(a,b,x,0,lamb)

    return soma

def E_y(y,lamb):

    soma = 0
    for a in [-1,1]:
        for b in [-1,1]:
            soma += b*p(a,b,1,y,lamb)

    return soma

# -----------------------------------------------

vertices = []

for a0 in [-1,1]:
    for a1 in [-1,1]:
        for b0 in [-1,1]:
            for b1 in [-1,1]:

                lamb = [a0, a1, b0, b1]

                ponto = []

                for x in range(2):
                    ponto.append(E_x(x,lamb))
                
                for y in range (2):
                    ponto.append(E_y(y,lamb))

                for x in range(2):
                    for y in range(2):
                        ponto.append(E(x,y,lamb))
                
                vertices.append(ponto)

def print_matrix(matrix):
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            print(matrix[i][j], end=' ')
        print()

print_matrix(vertices)   
