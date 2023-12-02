import gymnasium as gym
import matplotlib.pyplot as plt
import pathlib
from pathlib import Path
import datetime as dt
import numpy as np
import random
import time

#PARAMETROS

#El hugo y sus parametros ://////
iteracion = 5 #episodios
learning_rate = 0.3
intentos = 1 #numero de intentos
discount_rate = 0.4 #gamma
exploration_chance = 0.5 #epsilon
semilla = 56738
mininum_chance = 0.03
decreasing_decay = 0.01
rewards_per_iteracion = list()
env = gym.make('FrozenLake-v1', desc=None, render_mode="human" ,map_name="4x4", is_slippery=False) #ambiente
directorio = None

#Qtable Printout
qtable = np.zeros((16,4))
print(qtable)


#PARA GUARDAR LA TABLA
#algobn las tablas ya se ponen en el directorio
today = dt.datetime.now()
def plot_episode_rewards(rewards, directorio):
    
    if directorio is None:
        directorio = Path.cwd()
    
    print("Directorio: "+str(directorio))

    fig_directorio = directorio/'ml23'/'proyecto_final_QL'/'reward figures'


    timestamp = dt.datetime.now().strftime("%Y%m%d%H%M%S")
    timestamp = int(timestamp)
    filename = f'recompensas_{timestamp}.png'
    rew_file = fig_directorio/filename
    
    plt.plot(rewards)
    plt.title('Episode Rewards Over Time')
    plt.xlabel('Episode')
    plt.ylabel('Cumulative Reward')

    plt.savefig(rew_file)

    plt.show()
    
        
    

#Entrenamiento del Modelo
   
from gym.envs.toy_text.frozen_lake import generate_random_map #Utilizar esto para generar mapas y evaluar al agente entrenado
#metodo para entrenar al agente 
def training_waf(iteracion, intentos, exploration_chance, qtable, learning_rate, discount_rate, mininum_chance, decreasing_decay, rewards_per_iteracion):
    #Entrenamiento del agente imbezil
    #Iteracion sobre varios episodios :((
    e=0
    total_episode_reward = 0
    
    for e in range(iteracion):
        # Inicializa primer estado del episodio (actual)
        
        current_state = env.reset()[0] #GRAAAAAAAAAAAAAAH X_x
        done = False
    
        # Suma de las recompensas obtenidas del agente en el ambiente
        total_episode_reward = 0
    
        for i in range(intentos): 
            
            # Se selecciona un valor float por medio de una distribucion uniforme entre 0 y 1

            # Si el valor es menor a "exploration proba"
            #     El imbezil selecciona una accion aleatoria
            # else
            #     Explota su conocimiento utilizando la ecuacion de bellman
        
            if np.random.uniform(0,1) < exploration_chance:
                action = env.action_space.sample()
            else:
                action = np.argmax(qtable[current_state,:])
        
            # El ambiente corre la accion seleccionada y regresa lo siguiente:
            # El proximo estado, una recompensa y verdadero si el episodio se ha acabado
            info = env.step(action)
            next_state, reward, done, _, _ = info
        
            #Se actualiza la Q-table utilizando Q-learning iteration
            qtable[int(current_state), action] = (1-learning_rate) * qtable[int(current_state), action] +learning_rate*(reward + discount_rate*np.max(qtable[int(next_state),:]))
            total_episode_reward = total_episode_reward + reward
            # Si el episodio se ha acabado, salimos del loop
            if done:
                break
            print(f"iteracion: {e+1}")
            current_state = next_state
            env.render()
       
        print(info)
        
    
        # Se actualiza la probabilidad de exploracion utilizando la formula de exponential decay
        exploration_chance = max(mininum_chance, np.exp(-decreasing_decay*e))
        rewards_per_iteracion.append(total_episode_reward)   
    env.close()
    
# Llamar los metodos

training_waf(iteracion, intentos, exploration_chance, qtable, learning_rate, discount_rate, mininum_chance, decreasing_decay, rewards_per_iteracion)
plot_episode_rewards(rewards_per_iteracion, directorio)