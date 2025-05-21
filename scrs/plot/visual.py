import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

def plot_train_loss(train_losses, title="Train Loss", save_dir=None, filename="train_loss.png"):
    if save_dir is None:
        raise ValueError("Vous devez spécifier un répertoire de sauvegarde (save_dir).")
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, filename)

    plt.figure()
    plt.plot(train_losses, label='Train Loss', color='red', marker='x')
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(path)
    plt.close()
    print(f"Courbe de loss sauvegardée dans : {path}")

def plot_train_val_accuracy(train_accs, val_accs, title="Train vs Val Accuracy", save_dir=None, filename="train_val_accuracy.png"):
    if save_dir is None:
        raise ValueError("Vous devez spécifier un répertoire de sauvegarde (save_dir).")
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, filename)

    plt.figure()
    plt.plot(train_accs, label='Train Accuracy', marker='o')
    plt.plot(val_accs, label='Validation Accuracy', marker='s')
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig(path)
    plt.close()
    print(f"Courbe train/val accuracy sauvegardée dans : {path}")

def plot_evaluation(y_true, y_pred, labels=None, save_dir=None, cm_filename="confusion_matrix.png", report_filename="classification_report.txt"):
    """
    Affiche et sauvegarde :
    - la matrice de confusion (image PNG)
    - le rapport de classification (fichier texte)
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Matrice de confusion
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels)
    plt.xlabel("Prédit")
    plt.ylabel("Réel")
    plt.title("Matrice de confusion")
    cm_path = os.path.join(save_dir, cm_filename)
    plt.tight_layout()
    plt.savefig(cm_path)
    plt.close()
    print(f"Matrice de confusion sauvegardée : {cm_path}")
    
    # Rapport de classification
    report = classification_report(y_true, y_pred, target_names=labels if labels else None)

    report_path = os.path.join(save_dir, report_filename)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Rapport sauvegardé dans : {report_path}")