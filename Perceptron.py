import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import numpy as np
import copy


class Perceptron:
    def __init__(self, num_inputs, learning_rate=0.1):
        self.weights = [0.0 for _ in range(num_inputs)]
        self.bias = 0.0
        self.learning_rate = learning_rate

    def predict(self, inputs):
        activation = self.bias
        for i in range(len(inputs)):
            activation += self.weights[i] * inputs[i]
        return 1 if activation >= 0.0 else 0

    def train_step(self, inputs, actual_label):
        prediction = self.predict(inputs)
        error = actual_label - prediction

        if error != 0:
            for i in range(len(self.weights)):
                self.weights[i] += self.learning_rate * error * inputs[i]
            self.bias += self.learning_rate * error

        return error


# ==========================================
# 1. SETUP DATE
# ==========================================
training_data = [
    [1.0, 1.5], [2.0, 1.0],  # Sănătos (0)
    [4.0, 5.0], [5.0, 4.5]  # Bolnav (1)
]
labels = [0, 0, 1, 1]

x_coords = [point[0] for point in training_data]
y_coords = [point[1] for point in training_data]
colors = ['green' if label == 0 else 'red' for label in labels]

ai_engine = Perceptron(num_inputs=2, learning_rate=0.1)

# ==========================================
# 2. PRE-CALCULAREA ȘI SALVAREA ISTORICULUI
# ==========================================
history = []

# Salvăm starea 0 (înainte de a începe)
history.append({
    'weights': copy.deepcopy(ai_engine.weights),
    'bias': ai_engine.bias,
    'title': "Pas 0: Starea Inițială",
    'point_idx': None  # Niciun punct nu e selectat încă
})

epochs = 10
step_counter = 1

for epoch in range(epochs):
    total_errors = 0

    for i in range(len(training_data)):
        inputs = training_data[i]
        label = labels[i]

        error = ai_engine.train_step(inputs, label)
        if error != 0:
            total_errors += 1

        # Salvăm starea DUPĂ fiecare evaluare a unui punct
        history.append({
            'weights': copy.deepcopy(ai_engine.weights),
            'bias': ai_engine.bias,
            'title': f"Pas {step_counter} (Epoca {epoch + 1}) | Eroare pe calcul: {error != 0}",
            'point_idx': i  # Salvăm indexul punctului la care ne uităm
        })
        step_counter += 1

    if total_errors == 0:
        history.append({
            'weights': copy.deepcopy(ai_engine.weights),
            'bias': ai_engine.bias,
            'title': "GATA: Algoritmul a convergut perfect!",
            'point_idx': None
        })
        break

# ==========================================
# 3. INTERFAȚA VIZUALĂ INTERACTIVĂ
# ==========================================
current_step = 0

fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.2)  # Facem loc jos pentru butoane


def draw_step(step_idx):
    ax.clear()
    state = history[step_idx]

    # Desenăm punctele medicale
    ax.scatter(x_coords, y_coords, c=colors, s=100, edgecolors='black', zorder=2)

    # Evidențiem cu galben punctul evaluat în pasul curent
    if state['point_idx'] is not None:
        px, py = training_data[state['point_idx']]
        ax.scatter(px, py, c='yellow', s=180, edgecolors='orange', linewidth=2, zorder=1)

    # Desenăm linia
    w1 = state['weights'][0]
    w2 = state['weights'][1]
    b = state['bias']

    if w2 != 0:
        x_line = np.array([0, 6])
        y_line = -(w1 / w2) * x_line - (b / w2)
        ax.plot(x_line, y_line, color='blue', linestyle='--', linewidth=2, zorder=1)

    ax.set_xlim(0, 6)
    ax.set_ylim(0, 6)
    ax.set_title(state['title'])
    plt.draw()


# Funcțiile atașate butoanelor
def next_clicked(event):
    global current_step
    if current_step < len(history) - 1:
        current_step += 1
        draw_step(current_step)


def prev_clicked(event):
    global current_step
    if current_step > 0:
        current_step -= 1
        draw_step(current_step)


# Desenăm prima stare
draw_step(0)

# Creăm butoanele grafice din Matplotlib
axprev = plt.axes([0.3, 0.05, 0.15, 0.075])
axnext = plt.axes([0.55, 0.05, 0.15, 0.075])
btn_prev = Button(axprev, '<- Back')
btn_next = Button(axnext, 'Next ->')

# Conectăm butoanele la funcții
btn_prev.on_clicked(prev_clicked)
btn_next.on_clicked(next_clicked)

plt.show()