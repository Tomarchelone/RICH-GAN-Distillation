import torch
import lightgbm as lgbm
import numpy as np

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


import matplotlib.pyplot as plt

import seaborn as sns
sns.set()
sns.set_palette(sns.color_palette("muted"))

plt.rc('font', size=18)
plt.rc('axes', titlesize=18)
#plt.rc('axes', labelsize=MEDIUM_SIZE)
#plt.rc('xtick', labelsize=SMALL_SIZE)
#plt.rc('ytick', labelsize=SMALL_SIZE)
plt.rc('legend', fontsize=18)
plt.rc('figure', titlesize=18)

DLL_DIM = 5
INPUT_DIM = 3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dll_columns = ['RichDLLe', 'RichDLLk', 'RichDLLmu', 'RichDLLp', 'RichDLLbt']

def save_output_of_layer(model, input, output):
    model.saved = output

def plot_distributions(real, gen, w, epoch):
    fig, axes = plt.subplots(2, 2, figsize=(24, 12))

    for particle_type, ax in zip((0, 1, 3, 4), axes.flatten()):
        _, bins, _ = ax.hist(
            real[:, particle_type]
            , bins=100
            , label="real"
            , density=True
            , weights=w
        )

        ax.hist(
            gen[:, particle_type]
            , bins=bins
            , label="gen"
            , alpha=0.5
            , density=True
            , weights=w
        )

        ax.legend()
        ax.set_title(dll_columns[particle_type])

    plt.savefig(f"plots/teacher_{epoch}e.png", dpi=300, bbox_inches='tight')
    plt.show()

    # real vs teacher
    classifier = lgbm.LGBMClassifier()

    y_real = np.zeros(len(real))
    y_gen = np.ones(len(gen))

    y = np.concatenate((y_real, y_gen), axis=0)
    X = np.concatenate((real, gen), axis=0)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1889)

    classifier.fit(X_train, y_train)

    probs = classifier.predict_proba(X_test)

    print("[REAL VS TEACHER] ROC AUC:", roc_auc_score(y_test, probs[:, 1]))

def plot_metrics(real, teacher_gen, student_gen, w, info, savename):
    f = open(f"results/{savename}.txt", 'a')
    for key in info:
        f.write(f"{key}: {info[key]:.4f}\n")
    fig, axes = plt.subplots(2, 2, figsize=(24, 12))

    for particle_type, ax in zip((0, 1, 3, 4), axes.flatten()):
        _, bins, _ = ax.hist(
            real[:, particle_type]
            , bins=100
            , label="real"
            , density=True
            , weights=w
        )

        ax.hist(
            teacher_gen[:, particle_type]
            , bins=bins
            , label="teacher"
            , alpha=0.5
            , density=True
            , weights=w
        )

        ax.hist(
            student_gen[:, particle_type]
            , bins=bins
            , histtype='step'
            , label="student"
            , linewidth=2.0
            , color='black'
            , density=True
            , weights=w
        )

        ax.legend()
        ax.set_title(dll_columns[particle_type])

    fig.tight_layout()
    plt.savefig(f"plots/{savename}_{info['epoch']}e.png", dpi=300)
    plt.show()

    # real vs teacher
    classifier = lgbm.LGBMClassifier()

    y_real = np.zeros(len(real))
    y_teacher = np.ones(len(teacher_gen))

    y = np.concatenate((y_real, y_teacher), axis=0)
    X = np.concatenate((real, teacher_gen), axis=0)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1889)

    classifier.fit(X_train, y_train)

    probs = classifier.predict_proba(X_test)

    f.write(f"[REAL VS TEACHER] ROC AUC: {roc_auc_score(y_test, probs[:, 1]):.4f}\n")

    # real vs teacher
    classifier = lgbm.LGBMClassifier()

    y_real = np.zeros(len(real))
    y_student = np.ones(len(student_gen))

    y = np.concatenate((y_real, y_student), axis=0)
    X = np.concatenate((real, student_gen), axis=0)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1889)

    classifier.fit(X_train, y_train)

    probs = classifier.predict_proba(X_test)

    f.write(f"[REAL VS STUDENT] ROC AUC: {roc_auc_score(y_test, probs[:, 1]):.4f}\n")

    # teacher vs student
    classifier = lgbm.LGBMClassifier()

    y_teacher = np.zeros(len(teacher_gen))
    y_student = np.ones(len(student_gen))

    y = np.concatenate((y_teacher, y_student), axis=0)
    X = np.concatenate((teacher_gen, student_gen), axis=0)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1889)

    classifier.fit(X_train, y_train)

    probs = classifier.predict_proba(X_test)

    f.write(f"[TEACHER VS STUDENT] ROC AUC: {roc_auc_score(y_test, probs[:, 1]):.4f}\n\n")

    f.close()


def plot_metrics_exp(real, teacher_gen, student_gen, w, info, savename):
    # teacher vs student
    classifier = lgbm.LGBMClassifier()

    y_teacher = np.zeros(len(teacher_gen))
    y_student = np.ones(len(student_gen))

    y = np.concatenate((y_teacher, y_student), axis=0)
    X = np.concatenate((teacher_gen, student_gen), axis=0)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1889)

    classifier.fit(X_train, y_train)

    probs = classifier.predict_proba(X_test)

    teacher_vs_student = roc_auc_score(y_test, probs[:, 1])

    return teacher_vs_student
