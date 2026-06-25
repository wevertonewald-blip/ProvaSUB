from dataclasses import dataclass
import numpy as np
from scipy.integrate import solve_ivp

@dataclass
class paran:
  '''
  Parâmetros usados para a solução do modelo para a fabricação de compósitos poliméricos 

  Kr= Constantre cinética referencia para a solução do metodo de arrehnius (/h¹)
  m e n são parâmetros cinéticos usados, os quais são admensonais
  Ea= Energia de ativição para que a reação poça ocorrer (J/mol)
  R= Constante universla do gases(J/mol*K)
  Tr= Temperatura de referência (K)
  alpha0= Condição inicial para a cura de compósitos
  '''
  Kr: float = 0.25
  m: float = 0.8
  n: float = 1.5
  Ea: float = 45000.0
  R: float = 8.314
  Tr: float = 298.15
  alpha0: float = 0.05

def karr(T, paran):
  '''
  Nesse passo calcula-se a constante de arrhneius para a temperatura T em que o sistema se encontra
  '''
  k=paran.Kr*np.exp((-paran.Ea/paran.R)*((1/T)-(1/paran.Tr)))
  return k

def model(t, alpha, T, paran):
  '''
  Neste passo observa-se que há a impregação do modelo de cura dos copósitos, retornando a derivada da cura em função do tempo em que ela ficou curando 
  '''
  dalpha=karr(T, paran)*alpha**(paran.m)*(1-alpha)**(paran.n)
  return dalpha

def simu(temp, T, paran):
  '''
  Usa-se tal função  pra obter os resultados em que se quer simular.
  '''
  sol=solve_ivp(model, [0, temp], [paran.alpha0],
                args=(T, paran),
                t_eval=np.linspace(0, temp, 1000), method='RK45')
  return sol.t, sol.y[0]
