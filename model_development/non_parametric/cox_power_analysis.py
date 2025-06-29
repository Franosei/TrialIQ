import numpy as np
from statsmodels.stats.power import NormalIndPower
from math import log
from scipy.stats import norm


def required_events_for_cox_power(alpha=0.05, power=0.80, hr=1.5, allocation_ratio=0.5):
    """
    Calculate the number of events needed for Cox PH model with given power.
    Uses Freedman approximation.
    """
    z_alpha = norm.ppf(1 - alpha / 2)
    z_beta = norm.ppf(power)
    
    effect_size = log(hr)
    p = allocation_ratio

    n_events = ((z_alpha + z_beta)**2) / (effect_size**2 * p * (1 - p))
    return int(np.ceil(n_events))


def check_design_power(n_trials, event_rate, hr=1.5, alpha=0.05, power=0.80):
    n_events_actual = int(n_trials * event_rate)
    n_events_required = required_events_for_cox_power(alpha, power, hr)

    print("--- Cox Proportional Hazards Power Analysis ---")
    print(f"Target HR: {hr:.2f}, Alpha: {alpha}, Power: {power:.2f}")
    print(f"Sample size: {n_trials}, Event rate: {event_rate:.2%}")
    print(f"Required events: {n_events_required}, Actual events: {n_events_actual}")

    if n_events_actual >= n_events_required:
        print("Study is sufficiently powered.")
    else:
        print("Study is underpowered. Consider increasing sample or duration.")


if __name__ == "__main__":
    # Customize these values as needed
    total_trials = 97404
    event_rate = 0.15
    hazard_ratio = 1.5
    alpha = 0.05
    desired_power = 0.80

    check_design_power(total_trials, event_rate, hazard_ratio, alpha, desired_power)
