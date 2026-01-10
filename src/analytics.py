import numpy as np
from collections import defaultdict
from .config import PIXELS_PER_METER, FPS

def calculate_speed(p1, p2, fps=FPS, pixels_per_meter=PIXELS_PER_METER):
    """
    Calculate speed in km/h between two points.
    p1, p2: (x, y) tuples
    """
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    distance_pixels = np.sqrt(dx**2 + dy**2)
    distance_meters = distance_pixels / pixels_per_meter
    
    # Speed in m/s
    speed_mps = distance_meters * fps
    
    # Speed in km/h
    speed_kph = speed_mps * 3.6
    return speed_kph

def detect_ball_acquisition(player_positions, ball_position, threshold=30):
    """
    Determine which player has the ball.
    player_positions: dict {player_id: (x, y)}
    ball_position: (x, y) or None
    threshold: distance in pixels
    
    Returns: player_id or None
    """
    if not ball_position:
        return None
        
    closest_player = None
    min_dist = float('inf')
    
    for pid, pos in player_positions.items():
        if pos is None:
            continue
        dist = np.sqrt((pos[0] - ball_position[0])**2 + (pos[1] - ball_position[1])**2)
        if dist < min_dist:
            min_dist = dist
            closest_player = pid
            
    if min_dist < threshold:
        return closest_player
    return None

def summarize_player_stats(speed_data):
    """
    Summarize average speed for each player.
    speed_data: dict {player_id: list of (frame, speed)}
    """
    summary = {}
    for pid, speeds in speed_data.items():
        if speeds:
            # speeds is a list of tuples (frame_idx, speed_val)
            values = [s[1] for s in speeds]
            avg_speed = np.mean(values)
            max_speed = np.max(values)
            summary[pid] = {
                'avg_speed': avg_speed,
                'max_speed': max_speed
            }
    return summary
