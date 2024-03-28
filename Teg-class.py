import numpy as np

def getProbability(armies_attacker, armies_defender, armies_won, max_dice_defender):
    probabilities = {
        (1, 1, 1): 15/36,        # π111
        (1, 1, 0): 21/36,        # π110
        (1, 2, 1): 55/216,       # π121
        (1, 2, 0): 161/216,      # π120
        (1, 3, 1): 25/144,       # π131 Mio
        (1, 3, 0): 119/144,      # π130 Mio

        (2, 1, 1): 125/216,      # π211
        (2, 1, 0): 91/216,       # π210
        (2, 2, 2): 295/1296,     # π222
        (2, 2, 1): 420/1296,     # π221
        (2, 2, 0): 581/1296,     # π220
        (2, 3, 2): 0.12590021,   # π232
        (2, 3, 1): 0.25475823,   # π231
        (2, 3, 0): 0.61934156,   # π230

        (3, 1, 1): 855/1296,     # π311
        (3, 1, 0): 441/1296,     # π310
        (3, 2, 2): 2890/7776,    # π322
        (3, 2, 1): 2611/7776,    # π321
        (3, 2, 0): 2275/7776,    # π320
        (3, 3, 3): 0.1376028807, # π333
        (3, 3, 2): 0.2146990741, # π332
        (3, 3, 1): 0.2646604938, # π331
        (3, 3, 0): 0.3830375514, # π330
    }
    if armies_attacker > 3:
        armies_attacker = 3
    if armies_defender > max_dice_defender: # Regla de TEG vs RISK
        armies_defender = max_dice_defender
    return probabilities[(armies_attacker, armies_defender, armies_won)]

def getMatrixEntries(current_state, post_state, max_dice_defender):
    attacker_dices_current, defender_dices_current = current_state
    attacker_dices_post, defender_dices_post = post_state
    armies_won = defender_dices_current - defender_dices_post
    armies_lost = attacker_dices_current - attacker_dices_post
    if armies_won < 0  or armies_lost < 0 or (armies_lost + armies_won) != min(min(attacker_dices_current, 3), min(defender_dices_current, max_dice_defender)) or (armies_lost + armies_won > 3): # Regla de TEG vs RISK
        return 0
    else:
        return getProbability(attacker_dices_current, defender_dices_current, armies_won, max_dice_defender)




class TEG:
    def __init__(self, max_armies_atacker, max_armies_defender, max_dice_defender=3):
        self.max_armies_atacker = max_armies_atacker
        self.max_armies_defender = max_armies_defender
        self.max_dice_defender = max_dice_defender

        self.transient_states, self.absorbing_states = self.getStates()
        self.Q, self.R = self.getMatrixQ(), self.getMatrixR()
        self.S = self.getMatrixS()


    def getStates(self):
        transient_states = []
        absorbing_states = []
        for i in range(1, self.max_armies_atacker+1):
            for j in range(1, self.max_armies_defender+1):
                transient_states.append((i, j))
        for i in range(1, self.max_armies_atacker+1):
            absorbing_states.append((i, 0))
        for j in range(1, self.max_armies_defender+1):
            absorbing_states.append((0, j)) 
        return transient_states, absorbing_states
    
    def getMatrixQ(self):
        Q = np.zeros((self.max_armies_atacker * self.max_armies_defender, self.max_armies_atacker * self.max_armies_defender))
        for i, transient_state_1 in enumerate(self.transient_states):
            for j, transient_state_2 in enumerate(self.transient_states):
                Q[i, j] = getMatrixEntries(transient_state_1, transient_state_2, self.max_dice_defender)
        return Q
    
    def getMatrixR(self):
        R = np.zeros((self.max_armies_atacker * self.max_armies_defender, self.max_armies_atacker + self.max_armies_defender))
        for i, transient_state in enumerate(self.transient_states):
            for j, absorbing_state in enumerate(self.absorbing_states):
                R[i, j] = getMatrixEntries(transient_state, absorbing_state, self.max_dice_defender)
        return R
    
    def getMatrixS(self):
        I = np.identity(self.Q.shape[0])
        return np.matmul(np.linalg.inv(I - self.Q), self.R)
    
    def getWinningProbability(self, armies_attacker, armies_defender):
        i = self.transient_states.index((armies_attacker, armies_defender))
        return self.S[i][:armies_attacker].sum()
    
teg = TEG(100, 80)
print(teg.getWinningProbability(100,80))