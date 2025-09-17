import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# %%
def neutral_thrust_awa(l_d, eta):
    drag_lift = 1 / (l_d * eta)
    return np.arctan(drag_lift)


# eta = 0.7  # Adjust to estimate L/D for whole system, not just kite.
lift_drag = 5
eta = 0.7
awa = neutral_thrust_awa(lift_drag, eta)
print(
    f"l_d: {lift_drag}, eta: {eta}, neutral_thrust_awa: {awa:.2f} radians; {np.rad2deg(awa):.2f} degrees"
)


# %%
# Deterine twa from awa, boat speed and aws.  Use AWS, TWS and BS for lengths of sides of triangles.  Use AWSα, TWSα and BSα for opposite angles.
# TWA and AWA are represented/measured outside the triangle (typically).  AWA is TWSα.  AWSα is π - TWA. So TWA is π - AWSα
def twa(awa=0.2, boat_speed=10, aws=8):  # awa in radians
    """
    Calculate true wind angle (TWA) from apparent wind angle (AWA, degrees),
    boat speed (knots or m/s), and apparent wind speed (same units as boat_speed).
    Returns TWA in degrees (0 = bow, 180 = dead astern).
    """
    # Law of cosines for true wind speed
    tws = np.sqrt(aws**2 + boat_speed**2 - 2 * aws * boat_speed * np.cos(awa))
    # Law of sines for true wind angle
    # sin(AWSα) = aws * sin(TWSα)/TWS.  # Clamp to [-1, 1] to avoid domain errors
    sin_aws_alpha = np.clip((aws * np.sin(awa)) / tws)  # Using twsα = awa
    twa = np.pi - np.arcsin(sin_aws_alpha)
    return twa, tws


# %%
lift_drag = 5
eta = 0.7
awa = neutral_thrust_awa(lift_drag, eta)
boat_speed = 15
aws = 15
twa_rad, tws = twa(awa, boat_speed, aws)
print(
    f"boat_speed: {boat_speed}, l/d: {lift_drag}, eta: {eta}, awa: {np.rad2deg(awa):.1f} deg, aws: {aws:.0f},  tws: {tws:.1f}, TWA: {np.rad2deg(twa_rad):.1f} deg, CompTWA: {180-np.rad2deg(twa_rad):.1f}"
)
# %%
