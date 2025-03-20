# Outils-de-Traitement-de-Corpus

## TP

### Partie 1 | étude de cas CoNLL 2003 :

1. Quelle type de tâche propose CoNLL 2003 ?

    Language-Independent Named Entity Recognition.

2. Quel type de données y a-t-il dans CoNLL 2003 ?

    The CoNLL-2003 named entity data consists of eight files covering two languages: English and German. 
    For each of the languages there is : 
    - a training file : used for training learning methods.
    - a development file : used for tuning the parameters of the learning methods.
    - a test file : used for testing and evaluating different learning methods.
    - a large file with unannotated data

3. A quel besoin répond CoNLL 2003 ?

    Corpus ->  entities of four types: persons (PER), organizations (ORG), locations (LOC) and miscellaneous names (MISC)

4. Quels types de modèles ont été entraînés sur CoNLL 2003 ?

    **Maximum Entropy Model** :
    - Three systems used Maximum Entropy Models in isolation.
    - Two more systems used them in combination with other techniques.
    
    **Hidden Markov Models** were always used in combination with other learning techniques. 
    
    **Conditional Markov Models** is also applied for combining classifiers.
    
    Learning methods based on connectionist approaches ：
    - **robust risk minimization**, a Winnow technique. one employed this technique in a combination of learners. 
    - **Voted perceptrons** 
    - **recurrent neural network (Long Short-Term Memory)** for finding named entities.

    **AdaBoost.MH**

    **Memory-Based Learning**

    **Transformation-Based Learning**

    **Support Vector Machines, SVM**

    **Conditional Random Fields, CRF**

    system combination

5. Est un corpus monolingue ou multilingue ?

    Multilingue

### Partie 2 | projet:

1. Définissez les besoins de votre projet:

    - dans quel besoin vous inscrivez vous ?

        Corpus de commentaire -> sentiment tag (positive ou négative)
    
    - quel sujet allez vous traiter ?

        Identifier le sentiment de commentaire sur steam pour savoir si un joueur aime ce jeu ou pas
    
    - quel type de tâche allez vous réaliser ?

        Sentiment classification
    
    - quel type de données allez vous exploiter ?

        Je vais créer une tableu avec 4 coloums
        1. Nom de joueur
        2. heurs de jouer
        3. commentaires
        4. tag (recommend ou not recommend)

    - où allez vous récupérer vos données ?

        Sur le page de reviews de jeux
    
    - sont-elles libres d'accès ?

        Oui
