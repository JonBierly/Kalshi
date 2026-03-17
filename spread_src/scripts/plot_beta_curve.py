import joblib
import numpy as np
import matplotlib.pyplot as plt
import os

# Load the fitted calibrator
calibrator = joblib.load('models/beta_calibrator_v1.pkl')

# Generate a smooth range of inputs from 0 to 1
x = np.linspace(0.001, 0.999, 1000)

# Get the calibrated outputs
y = calibrator.predict(x)

# Plot the calibration mapping curve
plt.figure(figsize=(10, 8))
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', alpha=0.7, label='Identity (No Calibration)')
plt.plot(x, y, color='blue', linewidth=3, label='Beta Calibration Mapping')

# Highlight regions of adjustment
plt.fill_between(x, x, y, where=(y > x), color='green', alpha=0.1, label='Boosted Probabilities')
plt.fill_between(x, x, y, where=(y < x), color='red', alpha=0.1, label='Reduced Probabilities')

plt.title('Beta Calibrator Mapping Function', fontsize=16)
plt.xlabel('Raw Model Probability', fontsize=12)
plt.ylabel('Calibrated Probability', fontsize=12)
plt.legend(loc='lower right', fontsize=12)
plt.grid(True, alpha=0.3)

# Add parameters text
if hasattr(calibrator, 'map_'):
    a, b, m = calibrator.map_[-3:]
    plt.text(0.05, 0.95, f'Parameters:\na = {a:.4f}\nb = {b:.4f}\nm = {m:.4f}', 
             transform=plt.gca().transAxes, fontsize=12,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

os.makedirs('reports/calibration', exist_ok=True)
plt.savefig('reports/calibration/beta_mapping_curve.png', dpi=150, bbox_inches='tight')
print("Saved mapping curve to reports/calibration/beta_mapping_curve.png")
