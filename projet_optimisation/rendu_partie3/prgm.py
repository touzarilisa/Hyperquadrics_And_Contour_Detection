#importation des librairie
import numpy as np


#implementation de la fonction HyperQuadrique 
def fonction_HQ (x, y, A, B, C, gamma, Nh) :  
    
    mon_hq = 0
    
    for i in range (Nh) :
        mon_hq += (np.abs(A[i]  *  x + B[i]  *  y + C[i])) ** gamma[i] 
          
    return mon_hq - 1


#implementation de la fonction « inside - ouside »
def fonction_Fio(x, y, l) :
    
    Nh = np.size(l, 0)
    s = 0
    for i in range (Nh) :
        s += (l[i][0] * x + l[i][1] * y + l[i][2]) ** 4
        
    return s ** 0.25


#intermédiaire de calcule |Ax + By + C|^3 pour calcule de derive
def fonction_f(x, y, l) :    
    
    s = (x * l[0] + y * l[1] + l[2]) ** 3   
    
    return s


#definition de la dericvié de la fonction « inside - ouside » par rapport a x y
def fonction_dFio_xy(x, y, l) : 
    
    grad_0 = 0#initialisation du vecteur 
    grad_1 = 0#      gradient 
    
    for i in range(0, np.size(l, 0)) :
        trv_1 = l[i][0] * fonction_f(x, y, l[i])
        trv_2 = l[i][1] * fonction_f(x, y, l[i])        
        grad_0 = grad_0 + trv_1        
        grad_1 = grad_1 + trv_2   
        
    grad = np.array([grad_0, grad_1])
    a = fonction_Fio(x, y, l)     
    grad = grad  *  (a ** ( - 3)) 
      
    return grad


#definition de la dericvié de la fonction « inside - ouside » par rapport a A, B
def fonction_dFio_ab(x, y, l):
    
    grad = np.zeros(np.size(l, 0) * 3)
    #initialisation du vect gradient [3 * Nh]
    
    for i in range(0, np.size(l, 0)) :
        grad[3 * i] = x * fonction_f(x, y, l[i][:]) * fonction_Fio(x, y, l) ** ( - 3)
            #la derivé de la fonction « inside - ouside » par rapport a A
        grad[3 * i + 1] = y * fonction_f(x, y, l[i][:]) * fonction_Fio(x, y, l) ** ( - 3)
            #la derivé de la fonction « inside - ouside » par rapport a B 
        grad[3 * i + 2] = fonction_f(x, y, l[i][:]) * fonction_Fio(x, y, l) ** ( - 3)
            #la derivé de la fonction « inside - ouside » par rapport a C
            
    return grad


#Definition des Poids wi = 1 / ||dfio(xi, yi, lambda)||
def fonction_wi(x, y, l) :
    
    s = 0.1 #initialisation ||dfio(xi, yi, lambda)||
    grad = fonction_dFio_xy(x, y, l)
    
    for i in range(0, len(grad)) :        
        s = s + grad[i] ** 2
        
    return 1 / s
            
#La fonction des termes de pénalité 
def fonction_P(A, B, k1, k2, smax, smin) :
    
    u1 = (2 / (k1 * smax)) ** 2
    u2 = (2 / (k2 * smin)) ** 2
    
    s = (max(0, u1 - (A ** 2 + B ** 2))) ** 2 \
        +(max(0, (A ** 2 + B ** 2) - u2)) ** 2  
        
    return s

#Le gradient des termes de pénalité 
def fonction_dP(l, k1, k2, smax, smin) :
    
    u1 = (2 / (k1 * smax)) ** 2 #Definition des coefs u1 
    u2 = (2 / (k2 * smin)) ** 2 #et u2
    s = np.zeros(np.size(l, 0) * 3)
    
    for i in range(0, np.size(l, 0)) :
        s[3 * i] = 4 * l[i][0] * ( - max(0, u1 - (l[i][0] ** 2 + l[i][1] ** 2)) \
                + max(0, (l[i][0] ** 2 + l[i][1] ** 2) - u2))
        s[3 * i + 1] = 4 * l[i][1] * ( - max(0, u1 - (l[i][0] ** 2 \
                + l[i][1] ** 2)) + max(0, (l[i][0] ** 2 + l[i][1] ** 2) - u2))
        s[3 * i + 2] = 0 
        
    return s
 

#Le critere initiale de la phase 2 
def fonction_J(x_pts, y_pts, l, v, k1, k2, smax, smin) :
    
    N = len(x_pts)
    Nh = np.size((l, 0))
    j, j1, j2 = 0, 0, 0
    
    for i in range (0, N) :
        j1 = j1 + fonction_wi(x_pts[i], y_pts[i], l) \
            * (1 - fonction_Fio(x_pts[i], y_pts[i], l)) ** 2
            
    for i in range (0, Nh) :
        j2 = j2 + fonction_P(l[i][0], l[i][1], k1, k2, smax, smin)   
        
    j = j1 * 0.5 + v * j2
    
    return j
  
    
#Le gradient du critere initiale de la phase 2 
def fonction_dJ(x_pts, y_pts, l, v, k1, k2, smax, smin) :
    
    N = len(x_pts)
    g1 = np.zeros(np.size(l, 0) * 3)
    g2 = np.zeros(np.size(l, 0) * 3) 
    
    for i in range(N) :
        df = fonction_dFio_ab(x_pts[i], y_pts[i], l)        
        g1 = g1 - (1 - fonction_Fio(x_pts[i], y_pts[i], l)) \
            * fonction_wi(x_pts[i], y_pts[i], l) * df  
            
    g2 = fonction_dP(l, k1, k2, smax, smin) 
    g = g1 + v * g2
    
    return g


# Def de la matrice hessienne du terme de penalité P
def fonction_HP(l, k1, k2, smax, smin) :
        
    u1 = (2 / (k1 * smax)) ** 2
    u2 = (2 / (k2 * smin)) ** 2
    H = np.zeros((np.size(l, 0) * 3, np.size(l, 0) * 3)) 
    # Matrice[Nh * 3, Nh * 3]  qui contiendra les termes de HP
    a2 = np.zeros(np.size(l, 0) * 3) 
    # vecteur qui contiendra les derivé partielles  /  a A^2
    b2 = np.zeros(np.size(l, 0) * 3) 
    # vecteur qui contiendra les derivé partielles  /  a B^2
    ab = np.zeros(np.size(l, 0) * 3) 
    # vecteur qui contiendra les derivé partielles  /  a A * B
    
    for i in range(0, np.size(l, 0)) : 
        
        a2[i] = 4 * ( - max(0, u1 - (l[i][0] ** 2 + l[i][1] ** 2))\
                + max(0, (l[i][0] ** 2 + l[i][1] ** 2) - u2)) + 8 * l[i][0]\
                ** 2 * (np.sign(max(0, u1 - (l[i][0] ** 2 + l[i][1] ** 2))) \
                + np.sign(max(0, (l[i][0] ** 2 + l[i][1] ** 2) - u2)))
                # les derivé partielles  /  a A^2
        b2[i] = 4 * ( - max(0, u1 - (l[i][0] ** 2 + l[i][1] ** 2)) \
                + max(0, (l[i][0] ** 2 + l[i][1] ** 2) - u2)) + 8 * l[i][1] \
                ** 2 * (np.sign(max(0, u1 - (l[i][0] ** 2 + l[i][1] ** 2))) \
                + np.sign(max(0, (l[i][0] ** 2 + l[i][1] ** 2) - u2)))
                # les derivé partielles  /  a B^2
        ab[i] = 8 * l[i][0] * l[i][1] * (np.sign(max(0, u1 - (l[i][0] ** 2 \
                + l[i][1] ** 2))) + np.sign(max(0, (l[i][0] ** 2 + l[i][1] \
                ** 2) - u2)))
                # les derivé partielles  /  a A * B 
                
    for i in range(0, np.size(l, 0)) :                   
        H[3 * i][3 * i] = a2[i]    #  Remplissage de la
        H[3 * i + 1][3 * i] = ab[i]  # Matrice Hessiennes
        H[3 * i][3 * i + 1] = ab[i]  #       Hp
        H[3 * i + 1][3 * i + 1] = b2[i]#
        
    return H
       

# Def de la matrice hessienne du critére 𝐽
def fonction_HJ(x_pts, y_pts, l, v, k1, k2, smax, smin) :
    
    N = len(x_pts) #nombre de points
    Nh = np.size((l, 0)) #Nh
    H1 = np.zeros((np.size(l, 0) * 3, np.size(l, 0) * 3)) 
    # matrice qui contiendra la matrice Hessienne premiere partie je J
    H2 = np.zeros((np.size(l, 0) * 3, np.size(l, 0) * 3)) 
    #  matrice qui contiendra la matrice Hessienne de P
    deriv_rx = np.zeros(np.size(l, 0) * 3)
    
    for i in range (N) :
        
        df = fonction_dFio_ab(x_pts[i], y_pts[i], l)
        a = fonction_wi(x_pts[i], y_pts[i], l)              
        deriv_rx = a * df 
        
        for j in range (Nh * 3) :    
            H1[j] = H1[j] + df[j] * deriv_rx 
            # construction des termes de la premiere partie
            
    H2 = fonction_HP(l, k1, k2, smax, smin) #la matrice Hessienne de P 
    
    return H1 + v * H2 #le resultat final
    

#Implementation de La méthode de Levenberg - Marquardt 
def fonction_LM (x_pts, y_pts, l, v, k1, k2, smax, smin, precision, nmax) :
    
    Nh = np.size(l, 0)
    B = 0.01
    n = 0
    delta_l = np.ones(np.size(l, 0) * 3) #dl = 1
    I= np.zeros((3*Nh,3*Nh))
    np.fill_diagonal(I, 1)
    
    while (abs(np.max(delta_l)) > precision) and (n<nmax) : #critére d'arret        
        j = fonction_J(x_pts, y_pts, l, v, k1, k2, smax, smin)
        dj = fonction_dJ(x_pts, y_pts, l, v, k1, k2, smax, smin)
        hj = fonction_HJ(x_pts, y_pts, l, v, k1, k2, smax, smin)
        delta_l = np.linalg.solve((hj + B * I), -dj)
        delta_l = np.reshape(delta_l, (Nh, 3))  
        j_l = fonction_J(x_pts, y_pts, l + delta_l, v, k1, k2, smax, smin)
        
        while j_l >=  j :
            B = B * 10
            delta_l = np.reshape(np.linalg.solve((hj + B * I), -dj), (Nh, 3))           
            j_l = fonction_J(x_pts, y_pts, l + delta_l, v, k1, k2, smax, smin)
            
        B = 0.1 * B
        l = l + delta_l               
        n = n + 1 
        
    return l
        