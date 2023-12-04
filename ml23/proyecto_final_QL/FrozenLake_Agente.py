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
iteracion = 600 #episodios
learning_rate = 0.3
intentos = 100 #numero de intentos
discount_rate = 0.4 #gamma
exploration_chance = 0.5 #epsilon
semilla = 56738
mininum_chance = 0.2
decreasing_decay = 0.01
rewards_per_iteracion = list()
env = gym.make('FrozenLake-v1', desc=None, render_mode="human" ,map_name="4x4", is_slippery=False) #ambiente
directorio = None

#Qtable Printout
qtable = np.zeros((env.observation_space.n,env.action_space.n))
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
    
def dynamic_params(episodios):
    if episodios < 200 :
        return 0.5, 0.2 #Exploration, Min Chance
    elif episodios < 400:
        return 0.3, 0.1
    else:
        return 0.1, 0.05    
    

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
                #Aplicamos Heuristica Direccional para ayudar al agente cuando se atore y caiga en los hoyos de posicion 5 y 12
                current_position = np.argwhere(np.array(env.desc)==b'S')[0]
                if current_position[1] == 3 and current_position[0] not in [1,2]:
                    action = 2 # Intentamos empujar al agente a escoger irse a la derecha
                else:
                    action = np.argmax(qtable[current_state,:])
        
            # El ambiente corre la accion seleccionada y regresa lo siguiente:
            # El proximo estado, una recompensa y verdadero si el episodio se ha acabado
            info = env.step(action)
            next_state, reward, done, _, _ = info
        
            #Se actualiza la Q-table utilizando Q-learning iteration
            qtable[int(current_state), action] = (1-learning_rate) * qtable[int(current_state), action] +learning_rate*(reward + discount_rate*np.max(qtable[int(next_state),:]))
            total_episode_reward = total_episode_reward + reward
            print(qtable)
            # Si el episodio se ha acabado, salimos del loop
            if done:
                break
            print(f"iteracion: {e+1}")
            current_state = next_state
            env.render()
       
        print(info)
        print(qtable)
    
        # Se actualiza la probabilidad de exploracion utilizando la formula de exponential decay
        exploration_chance, mininum_chance = dynamic_params(e)
        exploration_chance = max(mininum_chance, np.exp(-decreasing_decay*e))
        rewards_per_iteracion.append(total_episode_reward)   
    env.close()
    
# Llamar los metodos

training_waf(iteracion, intentos, exploration_chance, qtable, learning_rate, discount_rate, mininum_chance, decreasing_decay, rewards_per_iteracion)
plot_episode_rewards(rewards_per_iteracion, directorio)