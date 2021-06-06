#%%
import pandas as pd
import numpy as np

#%%
def getAdditiveMatrix(calve, sire, dam, fill_none=False, sire_ind=None, dam_ind=None):
    n = int(max(*calve, *sire, *dam)) + 1 # does not take nan (np.nan takes nan)

    if fill_none:
        set_calve = set(calve)
        calve_add = []
        sire_add = []
        dam_add = []
        
        for i in range(n):
            if i not in set_calve and i != sire_ind and i != dam_ind:
                calve_add.append(i)
                sire_add.append(sire_ind)
                dam_add.append(dam_ind)

        calve = np.block([np.asarray(calve_add), calve])
        sire = np.block([np.asarray(sire_add), sire])
        dam = np.block([np.asarray(dam_add), dam])

    # print(calve)
    # print(sire)
    # print(dam)

    answer = np.zeros(shape=(n, n))
    name_to_ind = {calve[i]: i for i in range(len(calve))}

    for i in range(n):

        if i not in name_to_ind or \
            (np.isnan(sire[name_to_ind[i]]) and np.isnan(dam[name_to_ind[i]])):

            for j in range(i):
                answer[i][j] = 0
                answer[j][i] = 0
            
            answer[i][i] = 1

        elif np.isnan(sire[name_to_ind[i]]) or np.isnan(dam[name_to_ind[i]]):
            known_parent = dam if np.isnan(sire[name_to_ind[i]]) else sire

            for j in range(i):
                answer[i][j] = 0.5 * answer[j][int(known_parent[name_to_ind[i]])]
                answer[j][i] = answer[i][j]

            answer[i][i] = 1

        else:
            for j in range(i):
                answer[i][j] = 0.5 * (answer[j][int(sire[name_to_ind[i]])] + answer[j][int(dam[name_to_ind[i]])])
                answer[j][i] = answer[i][j]

            answer[i][i] = 1 + 0.5 * answer[int(sire[name_to_ind[i]])][int(dam[name_to_ind[i]])]


    # print(answer)
    return answer

def getZMatrix(observations, maximum=None):
    if maximum is None:
        maximum = np.max(observations) + 1

    length = len(observations)
    answer = np.zeros(shape=(length, maximum))

    for i in range(len(observations)):
        answer[i][observations[i]] = 1

    return answer

class AnimalModel:
    def __init__(self, data, name_x, name_y, name_calve, name_sire, name_dam, alpha):
        names = set(pd.concat([data[name_calve].dropna(), data[name_sire].dropna(), data[name_dam].dropna()]))

        if 'None' in names:
            names.remove('None')

        names = list(names)
        names.sort() #for easier reading

        self.name_to_ind = {name: i for i, name in zip(range(len(names)), names)}
        self.ind_to_name = {i: name for i, name in zip(range(len(names)), names)}

        self.name_to_ind['None'] = np.nan

        self.calve = data[name_calve].apply(lambda name: self.name_to_ind[name])
        self.sire = data[name_sire].apply(lambda name: self.name_to_ind[name])
        self.dam = data[name_dam].apply(lambda name: self.name_to_ind[name])

        self.X = pd.get_dummies(data[name_x])
        self.x_columns = self.X.columns
        self.X = self.X.to_numpy()

        self.y = data[name_y].to_numpy()

        self.A = getAdditiveMatrix(self.calve, self.sire, self.dam)
        self.Z = getZMatrix(self.calve)

        self.alpha = alpha

    def calculate(self):
        X_t = self.X.T
        Z_t = self.Z.T

        X_tZ = X_t @ self.Z
        Z_tX = Z_t @ self.X
        X_tZ = Z_tX.T

        X_tX = X_t @ self.X
        Z_tZ = Z_t @ self.Z

        MME = np.block([[X_tX, X_tZ],
                        [Z_tX, Z_tZ + self.alpha * np.linalg.inv(self.A)]])

        X_ty = X_t @ self.y
        Z_ty = Z_t @ self.y

        y_new = np.block([X_ty, Z_ty])
        ba = np.linalg.inv(MME) @ y_new

        self.b = ba[:len(self.x_columns)]
        self.a = ba[len(self.x_columns):]

    def print_results(self):
        for i in range(len(self.x_columns)):
            print(self.x_columns[i], self.b[i])

        for i in self.ind_to_name:
            print(self.ind_to_name[i], self.a[i])

def getRmatrix(calve, sire, dam, sigma_e, sigma_a):
    sire = set(sire)
    dam = set(dam)

    answer = np.zeros(shape=(len(calve), len(calve)))

    diag = []
    for name in calve:
        if name in sire or name in dam:
            diag.append(sigma_e)
        else:
            diag.append(sigma_e + 1/2 * sigma_a)

    np.fill_diagonal(answer, diag)
    return answer    

def getWmatrix(calve, sire, dam):
    n = int(max(*sire, *dam)) + 1
    length = len(calve)
    answer = np.zeros(shape=(length, n))

    sire_set = set(sire)
    dam_set = set(dam)

    for name in range(length):
        if calve[name] in sire_set or calve[name] in dam_set:
            answer[name][calve[name]] = 1
            continue

        if not np.isnan(sire[name]):
            answer[name][int(sire[name])] = 0.5

        if not np.isnan(dam[name]):
            answer[name][int(dam[name])] = 0.5
    
    return answer


class ReducedAnimalModel:
    def __init__(self, data, name_x, name_y, name_calve, name_sire, name_dam, sigma_e, sigma_a):
        names = set(pd.concat([data[name_calve].dropna(), data[name_sire].dropna(), data[name_dam].dropna()]))

        if 'None' in names:
            names.remove('None')

        names = list(names)
        names.sort() #for easier reading

        self.name_to_ind = {name: i for i, name in zip(range(len(names)), names)}

        self.name_to_ind['None'] = np.nan

        self.calve = data[name_calve].apply(lambda name: self.name_to_ind[name])
        self.sire = data[name_sire].apply(lambda name: self.name_to_ind[name])
        self.dam = data[name_dam].apply(lambda name: self.name_to_ind[name])

        set_sire = set(data[name_sire])
        set_dam = set(data[name_dam])
        indexes = data[name_calve].apply(lambda x: x in set_sire or x in set_dam)

        self.A = getAdditiveMatrix(self.calve[indexes], self.sire[indexes], self.dam[indexes])

        self.ind_to_name = {i: name for i, name in zip(range(len(names)), names) if name in set_sire or name in set_dam}


        self.X = pd.get_dummies(data[name_x])
        self.x_columns = self.X.columns
        self.X = self.X.to_numpy()

        self.y = data[name_y].to_numpy()

        self.R = getRmatrix(self.calve, self.sire, self.dam, sigma_e, sigma_a)
        self.W = getWmatrix(self.calve, self.sire, self.dam)


        self.sigma_e = sigma_a
        self.sigma_a = sigma_a

    def calculate(self):
        X_t = self.X.T
        W_t = self.W.T
        R_inv = np.zeros(shape=self.R.shape)
        np.fill_diagonal(R_inv, 1 / np.diag(self.R))

        X_tR_inv = X_t @ R_inv
        W_tR_inv = W_t @ R_inv

    
        MME = np.block([[X_tR_inv @ self.X, X_tR_inv @ self.W],
                        [W_tR_inv @ self.X, W_tR_inv @ self.W + np.linalg.inv(self.A) / self.sigma_a]])

        
        y_new = np.block([X_tR_inv @ self.y, W_tR_inv @ self.y])
        ba = np.linalg.inv(MME) @ y_new

        self.b = ba[:len(self.x_columns)]
        self.a = ba[len(self.x_columns):]

    def print_results(self):
        for i in range(len(self.x_columns)):
            print(self.x_columns[i], self.b[i])

        for i in self.ind_to_name:
            print(self.ind_to_name[i], self.a[i])

#%%
if __name__ == '__main__':
    table = pd.DataFrame([["4", 'Male', "1", "None", 4.5],\
                        ["5", 'Female', "3", "2", 2.9],\
                        ["6", 'Female', "1", "2", 3.9],\
                        ["7", 'Male', "4", "5", 3.5],
                        ["8", 'Male', "3", "6", 5.0]], 
                        columns=['Calves', 'Sex', 'Sire', 'Dam', 'WWG'])

    animal = AnimalModel(table, 'Sex', 'WWG', 'Calves', 'Sire', 'Dam', 2)
    animal.calculate()
    animal.print_results()

#%%
if __name__ == '__main__':
    table = pd.DataFrame([['Male', "1", "None", "None", 4.5],
                      ['Female', "3", "None", "None", 2.9],
                      ['Female', "1", "None", "None", 3.9],
                      ['Male', "4", "1", "None", 3.5],
                      ['Male', "3", "None", "None", 5]], 
                      columns=['Sex', 'Sire', 'Sire of sire', 'Dam of sire', 'WWG'])

    sire = AnimalModel(table, 'Sex', 'WWG', 'Sire', 'Sire of sire', 'Dam of sire', 11)
    sire.calculate()
    sire.print_results()

#%%
if __name__ == '__main__':
    table = pd.DataFrame([["4", 'Male', "1", "None", 4.5],\
                        ["5", 'Female', "3", "2", 2.9],\
                        ["6", 'Female', "1", "2", 3.9],\
                        ["7", 'Male', "4", "5", 3.5],
                        ["8", 'Male', "3", "6", 5.0]], 
                        columns=['Calves', 'Sex', 'Sire', 'Dam', 'WWG'])

    animal = ReducedAnimalModel(table, 'Sex', 'WWG', 'Calves', 'Sire', 'Dam', 40, 20)
    animal.calculate()
    animal.print_results()

# %%

class GroupAnimalModel:
    def __init__(self, data, name_x, name_y, name_calve, name_sire, name_dam, alpha):
        data = data.copy()
        data[name_sire] = data[name_sire].replace('None', 'Group1')
        data[name_dam] = data[name_dam].replace('None', 'Group2')

        names = set(pd.concat([data[name_calve].dropna(), data[name_sire].dropna(), data[name_dam].dropna()]))
        names.add('Group1')
        names.add('Group2')

        names = list(names)
        names.sort() #for easier reading

        self.name_to_ind = {name: i for i, name in zip(range(len(names)), names)}
        self.ind_to_name = {i: name for i, name in zip(range(len(names)), names)}


        self.calve = data[name_calve].apply(lambda name: self.name_to_ind[name])
        self.sire = data[name_sire].apply(lambda name: self.name_to_ind[name])
        self.dam = data[name_dam].apply(lambda name: self.name_to_ind[name])

        self.X = pd.get_dummies(data[name_x])
        self.x_columns = self.X.columns
        self.X = self.X.to_numpy()

        self.y = data[name_y].to_numpy()

        self.A = getAdditiveMatrix(self.calve, self.sire, self.dam, \
            fill_none=True, sire_ind=self.name_to_ind['Group1'], dam_ind=self.name_to_ind['Group2'])

        self.Z = getZMatrix(self.calve, maximum=len(names))

        self.alpha = alpha

    def calculate(self):
        X_t = self.X.T
        Z_t = self.Z.T

        X_tZ = X_t @ self.Z
        Z_tX = Z_t @ self.X
        X_tZ = Z_tX.T

        X_tX = X_t @ self.X
        Z_tZ = Z_t @ self.Z

        MME = np.block([[X_tX, X_tZ],
                        [Z_tX, Z_tZ + self.alpha * np.linalg.inv(self.A)]])
        
        print(np.linalg.inv(self.A))

        X_ty = X_t @ self.y
        Z_ty = Z_t @ self.y

        y_new = np.block([X_ty, Z_ty])
        ba = np.linalg.inv(MME) @ y_new

        self.b = ba[:len(self.x_columns)]
        self.a = ba[len(self.x_columns):]

    def print_results(self):
        for i in range(len(self.x_columns)):
            print(self.x_columns[i], self.b[i])

        for i in self.ind_to_name:
            print(self.ind_to_name[i], self.a[i])


# %%
table = pd.DataFrame([["4", 'Male', "1", "None", 4.5],\
                    ["5", 'Female', "3", "2", 2.9],\
                    ["6", 'Female', "1", "2", 3.9],\
                    ["7", 'Male', "4", "5", 3.5],
                    ["8", 'Male', "3", "6", 5.0]], 
                    columns=['Calves', 'Sex', 'Sire', 'Dam', 'WWG'])

animal = GroupAnimalModel(table, 'Sex', 'WWG', 'Calves', 'Sire', 'Dam', 2)
animal.calculate()
animal.print_results()


# %%

# %%
