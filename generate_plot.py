import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Set the style to be dark and modern to match the UI
plt.style.use('dark_background')

# 1. Generate hypothetical historical data (e.g. 36 months)
np.random.seed(42)
months = pd.date_range(start='2021-01-01', periods=36, freq='ME')
# Create a random walk with a downward trend to simulate improving retention
historical_rates = 25.0 + np.cumsum(np.random.normal(-0.15, 0.8, 36))
historical_rates = np.clip(historical_rates, 5.0, 35.0)

# 2. Generate hypothetical forecast data (next 12 months)
future_months = pd.date_range(start=months[-1], periods=13, freq='ME')
# Continuing the trend but with wider confidence intervals
forecast_rates = historical_rates[-1] + np.cumsum(np.random.normal(-0.2, 0.5, 13))
forecast_rates = np.clip(forecast_rates, 5.0, 35.0)

# Confidence interval bounds
upper_bound = forecast_rates + np.linspace(0, 3.5, 13)
lower_bound = forecast_rates - np.linspace(0, 3.5, 13)

# 3. Create the plot
fig, ax = plt.subplots(figsize=(10, 5), dpi=120)
fig.patch.set_facecolor('#0f172a') # Tailwind slate-900 background
ax.set_facecolor('#0f172a')

# Plot historical
ax.plot(months, historical_rates, color='#38bdf8', linewidth=2.5, label='Historical Dropout Rate')

# Plot forecast
ax.plot(future_months, forecast_rates, color='#f472b6', linewidth=2.5, linestyle='--', label='ARIMA Forecast')

# Fill confidence interval
ax.fill_between(future_months, lower_bound, upper_bound, color='#f472b6', alpha=0.15, label='95% Confidence Interval')

# Customize axes and grid
ax.grid(True, color='white', alpha=0.05)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('#475569')
ax.spines['bottom'].set_color('#475569')
ax.tick_params(colors='#94a3b8')

ax.set_ylabel('Dropout Rate (%)', color='#94a3b8', fontsize=11, labelpad=10)
ax.set_title('Hypothetical Student Dropout Forecast (Simulated Data)', color='#f8fafc', fontsize=14, pad=15)

# Add a vertical line where the forecast begins
ax.axvline(x=months[-1], color='#64748b', linestyle=':', alpha=0.5)
ax.text(months[-1], 27, ' Forecast Start', color='#94a3b8', fontsize=10)

# Legend
leg = ax.legend(loc='upper right', frameon=True, facecolor='#1e293b', edgecolor='#334155')
for text in leg.get_texts():
    text.set_color('#f8fafc')

plt.tight_layout()

# 4. Save the plot to the React public directory
output_path = '/home/chandima-bandara/Desktop/student-retention-analysis/frontend/public/forecast.png'
plt.savefig(output_path, facecolor=fig.get_facecolor(), edgecolor='none')
print(f"Plot saved successfully to {output_path}")
