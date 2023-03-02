import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.manifold import TSNE
from PIL import Image
from sklearn.neural_network import MLPClassifier

#############################################
# 1.2 Chargement des données d’entraînement #
#############################################

# question 1.2.1
train_data = pd.read_csv('digits.csv')

# question 1.2.2
# print(train_data.head(10))

#######################################
# 1.3 Pré-traitement et visualisation #
#######################################

# question 1.3.1
# plt.imshow(train_data.iloc[0, 0:-1].values.reshape(8, 8), cmap='gray')
# plt.show()

# question 1.3.2
features = train_data.columns[0:-1]
# print(features)

###############################
# 1.4 Visualisation avec TSNE #
###############################

# question 1.4.1
tsne = TSNE(n_components=2, random_state=42)
X_embedded = tsne.fit_transform(train_data[features].values)
print(X_embedded.shape)

# question 1.4.2
# fig = px.scatter(X_embedded, x=0, y=1, color=train_data['classe'])
# fig.show()

# question 1.4.3
# on observe que les points sont bien séparés mais y a des points ou il
# dont plus proche que d'autres numero vue la resemblence entre queqlues
# numero

# question 1.4.4
# on peut en déduire que les chances de succès sont assez bonne vu que les
# points sont bien séparés

##############################
# 1.5 Séparation des données #
##############################

# question 1.5.1
digits_train = train_data.sample(frac=0.8, random_state=90)
digits_valid = train_data.drop(digits_train.index)

# question 1.5.2
X_train = digits_train[digits_train.columns[:-1]]
Y_train = digits_train[digits_train.columns[-1]]
X_valid = digits_valid[digits_valid.columns[:-1]]
Y_valid = digits_valid[digits_valid.columns[-1]]


#################################
# 2 KPPV : K plus proches voisins #
#################################

# question 2.1
def euclidian_distance(v1, v2):
    distance = 0
    for i in range(v1.shape[0]):
        distance += (v1[i] - v2[i]) ** 2
    return np.sqrt(distance)


# question 2.2
def neighbors(X_train, y_label, x_test, k):
    list_distances = []
    # On calcule toute les distances entre x_test et chaque point de X_train.
    for i in range(X_train.shape[0]):
        distance = euclidian_distance(X_train.iloc[i], x_test)
        list_distances.append(distance)

    # On trie les poinds de X_train par ordre croissant de distance et on renvoie le dataframe qui contient les k plus proches voisins.
    df = pd.DataFrame()
    df["label"] = y_label
    df["distance"] = list_distances
    df = df.sort_values(by="distance")
    return df.iloc[:k, :]  # on renvoie les k premières lignes


# question 2.3
# plt.imshow(np.array(X_valid.iloc[3]).reshape(8, 8), cmap='Greys')
# plt.show()


# question 2.4
def prediction(neighbor):
    count = neighbor["label"].count()
    # renvoie le nom de label de la classe majoritaire
    return neighbor["label"].value_counts().index[0]


print(neighbors(X_train, Y_train, X_valid.iloc[0], 5))
print(prediction(neighbors(X_train, Y_train, X_valid.iloc[0], 5)))
print(Y_valid.iloc[0])


# question 2.7
def evaluation(X_train, Y_train, X_valid, Y_valid, k, verbose=True):
    TP = 0  # vrai
    FP = 0  # faux
    total = 0
    for i in range(X_valid.shape[0]):
        nearest_neighbors = neighbors(X_train, Y_train, X_valid.iloc[i], k)

        if prediction(nearest_neighbors) == Y_valid.iloc[i]:
            TP += 1
        else:
            FP += 1
            print("Erreur de prediction : ")
            print("Prediction : ", prediction(nearest_neighbors))
            print("Resultat attendu : ", Y_valid.iloc[i])
        total += 1

    accuracy = TP / total
    print("TP : ", str(TP))

    if verbose:
        print("Accuracy:" + str(accuracy))

    return accuracy


# question 2.8
# print(evaluation(X_train, Y_train, X_valid, Y_valid, 5));

# question 2.9
# list_accuracy = []
# for k in range(1, 19, 2):  # range de 1 à 3 avec un pas de 2
#    list_accuracy.append(evaluation(X_train, Y_train, X_valid, Y_valid, k, verbose=False))

# print(list_accuracy)

# question 2.10
# fig = px.line(x=range(1, 19, 2), y=list_accuracy, labels={'x': 'k', 'y': 'accuracy'})
# fig.show()

######################################################
# 3 Application du modèle sur des photos de chiffres #
######################################################

# question 3.1

mon_chiffre = [0, 0, 3, 13, 14, 2, 0, 0, 0, 2, 15, 14, 13, 15, 4, 0, 0, 11, 16, 4, 6, 12, 15, 1, 1, 16, 12, 0, 0, 6, 16,
               4, 4, 16, 8, 0, 0, 6, 16, 6, 4, 16, 10, 0, 0, 11, 16, 3, 0, 13, 15, 8, 10, 16, 12, 0, 0, 2, 12, 16, 16,
               12, 2, 0, 0]
nearest_neighbors = neighbors(X_train, Y_train, mon_chiffre, 5)
print(prediction(nearest_neighbors))

########################
# 4 Réseau de neurones #
########################

# question 4.1
clf = MLPClassifier(solver='adam', hidden_layer_sizes=350, alpha=0.0001, learning_rate='adaptive')
clf.fit(X_train, Y_train)


def evaluation(X_train, Y_train, X_valid, Y_valid, k, nn, verbose=True):
    TP = 0  # vrai
    FP = 0  # faux
    total = 0
    if nn == True:
        pred = clf.predict(X_valid)
        for i in range(X_valid.shape[0]):
            if pred[i] == Y_valid.iloc[i]:
                TP += 1
            else:
                print("Prediction : ", pred[i])
                print("Resultat attendu : ", Y_valid.iloc[i])
            total += 1
    accuracy = TP / total
    if verbose:
        print("Accuracy:" + str(accuracy))
    return accuracy


evaluation(X_train, Y_train, X_valid, Y_valid, 5, True, verbose=True)
