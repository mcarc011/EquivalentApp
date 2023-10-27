import numpy as np
import streamlit as st
import itertools
import time


def TupFind(L):
	maps = []
	n = 0
	while n!=len(L)-1:
		mt = []
		for m in range(n,len(L)):
			if n!=m:
				mt += [(n,m)]
		if mt != []:
			maps += [mt]
		n+=1
	return list(itertools.product(*maps))


def Swap(M:np.array, t:tuple):
	Mt = M.copy()
	Mt[t[0]],Mt[t[1]] = M[t[1]],M[t[0]]
	Mt = np.transpose(Mt)
	Mc = Mt.copy()
	Mt[t[0]], Mt[t[1]] = Mc[t[1]], Mc[t[0]]
	return np.transpose(Mt)


def equivalent(X1, F1, X2, F2,counter=False):
    if np.array_equal(X1,X2) and np.array_equal(F1,F2):
        return True

    ANodes = [(sorted(X1[node]),sorted(np.transpose(X1)[node]),sorted(F1[node])) for node in range(len(X1))]
    BNodes = [(sorted(X2[node]),sorted(np.transpose(X2)[node]),sorted(F2[node])) for node in range(len(X1))]

    try:
        Xt, Ft = X1.copy(), F1.copy()
        for i in range(len(X1)):
            if BNodes[i] != ANodes[i]:
                b = BNodes[i]
                j = ANodes[i:].index(b)+i
                Xt,Ft = Swap(Xt,(j,i)),Swap(Ft,(j,i))
                ANodes = [(sorted(Xt[node]),sorted(np.transpose(Xt)[node]),sorted(Ft[node])) for node in range(len(X1))]
    except:
        return False 

    if np.array_equal(Xt,X2) and np.array_equal(Ft,F2):
        return True
    
    #final test
    Aswaps = {}
    ANodes = [(sorted(Xt[node]),sorted(np.transpose(Xt)[node]),sorted(Ft[node])) for node in range(len(X1))]
    for i,a in enumerate(ANodes):
        for j,b in enumerate(ANodes):
            if a==b and i!=j:
                if str(a) not in Aswaps:
                    Aswaps[str(a)] = [i,i,j]
                if i not in Aswaps[str(a)]:
                    Aswaps[str(a)] += [i]
                if j not in Aswaps[str(a)]:
                    Aswaps[str(a)] += [j]

    temptylist = [val for val in Aswaps.values()]
    temptlist = [] 
    for tem in temptylist:
        mapto = TupFind(np.array(tem))
        tp = [[[0,0] for m in mapto[0]]]  
        for mapper in mapto:
            tp += [[[tem[m[0]],tem[m[1]]] for m in mapper]]
        temptlist += [tp]

    for tlist in itertools.product(*temptlist):
        Xp,Fp = Xt.copy(),Ft.copy()
        for tlistswap in itertools.permutations(tlist):
            for titer in tlistswap: 
                for step in titer:
                    t = (step[0],step[1])
                    Xp,Fp = Swap(Xp,t),Swap(Fp,t)
                    if np.array_equal(Xp,X2) and np.array_equal(Fp,F2):
                        return True
    return False

col1,col2 = st.columns(2)

st.title('Matrix Compare')

with col1:
    matrix1 = st.text_area('Matrix 1')
    compare = st.button('Compare')

with col2:
    matrix2 = st.text_area('Matrix 2')

if compare:
    m1 = eval(matrix1)
    m2 = eval(matrix2)
    results = equivalent(m1,m1,m2,m2)
    with col1:
        st.write(results)

