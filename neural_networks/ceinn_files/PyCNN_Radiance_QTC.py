# IMPORTAÇÃO DAS BIBLIOTECAS
import scipy.signal as sig
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
from scipy.integrate import solve_ivp
from tqdm import tqdm


# INICIALIZAÇÃO DO INSTRUMENTO
canais = [12]
sat = "N19"
sensor = "AMSUA"
gdhs = ["2023020100"]
for gdh in gdhs:
    for canal in canais:
        print(f'Rodada para o canal {canal}')

        # CONVERSÃO DA IMAGEM P/ ESCALA DE CINZA E DEFINIÇÃO DAS DIMENSÕES 
        arq_fg = np.loadtxt(f'/home/cesar/ges_ch_{canal}')
        arq_obs = np.loadtxt(f'/home/cesar/obs_ch_{canal}')
        image = (np.array(arq_fg))
        image2 = (np.array(arq_obs))
        # MÁSCARAS FEEDBACK('AY') e CONTROL('BU')
        feedback_template = [0,0,0]

        # MANIPULAÇÃO DA MATRIZ DA IMAGEM
        mat_x = image
        mat_y = image2

        height_i = len(image)

        # ENTRADAS (Bias, Epocas e Condição Inicial)
        Bias = 0
        EPC = 50
        CI = 1

        # MANIPULAÇÃO DA CONDIÇÃO INICIAL E AJUSTE PARA DEFINIR 'BU'
        U = mat_x
        z0 = mat_x * CI

        # ABERTURA DAS MATRIZES 
        L1=[]
        l1=[]
        L2=[]
        l2=[]
        L3=[]
        l3=[]
        X=[]
        DX=[]
        linha_x=[]
        linha_dx=[]
        A=[]
        B=[]
        lista_DV = []
        Cont = []
        Cost = [0]
        # DERIVADA
        funcao_deriv = lambda a, b, x, y, w, z: -x + y + w + z
        lim = mat_y - z0
        # LOOPING PASSANDO POR CADA PIXEL EM 'T' EPOCAS
        for t in tqdm(range(0,EPC,1)):
            interup2 = mat_y - z0
            for i in range(0,height_i, 1):
               
                # DEFINIÇÃO DAS CONDIÇÕES P/ BORDAS DA IMAGEM
                if i==0:
                    #print(i,j,'canto superior esquerdo')
                    x2 = 0
                    x3 = U[i]
                    x4 = U[i+1]
                    obs2 = 0
                    obs3 = mat_y[i]
                    obs4 = mat_y[i+1]

                elif i==height_i-1:
                    #print(i,j,'linha inferior')
                    x2 = U[i-1]
                    x3 = U[i]
                    x4 = 0
                    obs2 = mat_y[i-1]
                    obs3 = mat_y[i]
                    obs4 = 0

                else:
                    #print(i,j)
                    x2 = U[i-1]
                    x3 = U[i]
                    x4 = U[i+1]
                    obs2 = mat_y[i-1]
                    obs3 = mat_y[i]
                    obs4 = mat_y[i+1]

                if i==0:
                    #print(i,j,'canto superior esquerdo')
                    control_template = [ 0, (mat_y[i]/z0[i])/2, (mat_y[i]/z0[i+1])/2]
                    z2 = 0
                    z3 = z0[i]  
                    z4 = z0[i+1]

                elif i==height_i-1:
                    #print(i,j,'linha inferior')
                    control_template = [(mat_y[i]/z0[i-1])/2, (mat_y[i]/z0[i])/2, 0]
                    z2 = z0[i-1]
                    z3 = z0[i]
                    z4 = 0

                else:
                    #print(i,j)
                    control_template = [(mat_y[i]/z0[i-1])/3, (mat_y[i]/z0[i])/3, (mat_y[i]/z0[i+1])/3]
                    z2 = z0[i-1]
                    z3 = z0[i]
                    z4 = z0[i+1]

                # FUNÇÃO DE ATIVAÇÃO PARA 'AY'
                
                y2= 0.5 * (abs((z2 - obs2) + 1) - abs((z2 - obs2) - 1))
                y3= 0.5 * (abs((z3 - obs3) + 1) - abs((z3 - obs3) - 1))
                y4= 0.5 * (abs((z4 - obs4) + 1) - abs((z4 - obs4) - 1))

                # MONTAGEM DAS MATRIZES DE VIZINHANÇA Y E U

                A.append(-y2)
                A.append(-y3)
                A.append(-y4)

                B.append(x2)
                B.append(x3)
                B.append(x4)


                # DEFINIÇÃO 'AY' E 'BU'
                Ay = A[0]*feedback_template[0] + A[1]*feedback_template[1] + A[2]*feedback_template[2] 
                Bu = B[0]*control_template[0] + B[1]*control_template[1] + B[2]*control_template[2]

                # CALCULO DA DERIVADA PONTO A PONTO
                x_deriv = funcao_deriv(0,0,z3,Bias,Ay, Bu)
                # CALCULO DA INTEGRAL PONTO A PONTO
                x_integ = solve_ivp(funcao_deriv, [0,1], [z3], args=(z3,Bias,Ay, Bu), method='RK45')
                # MONTAGEM DA LINHA DA MATRIZ            
                x_novo = float(x_integ.y[0][1])

                A=[]
                B=[]

                # MONTAGEM DA MATRIZ
                X.append(x_novo)
                DX.append(x_deriv)

            # VERIFICANDO SE DX ESTÁ CONVERGINDO PARA 0. CASO NÃO ESTEJA OBEDECENDO A CONDIÇÃO, VOLTA PARA MAIS UMA ÉPOCA DE LOOP.
            z0 = (np.array(X))
            DX = (np.array(DX))
       
            X = []
            DX = []
        
        # NOVO ESTADO DA FÍGURA ENTRANDO NA FUNÇÃO DE ATIVAÇÃO
        mat_x = z0
        np.savetxt(f'/home/cesar/cnn_ch_{canal}',mat_x,fmt='%s')
