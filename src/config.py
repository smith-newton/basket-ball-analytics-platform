
# Team configurations
TEAM_A = {
    'class_id': 4,
    'color': (238, 130, 238),  # Violet
    'name': 'Team A',
    'id_range': range(1, 6) # IDs 1-5
}

TEAM_B = {
    'class_id': 5,
    'color': (0, 0, 255),      # Red
    'name': 'Team B',
    'id_range': range(6, 11)    # IDs 6-10
}

# New class IDs mapping
NEW_COACH_CLASS_ID = 1
NEW_COACH_ID_RANGE = range(11, 13)

NEW_HOOP_CLASS_ID = 2
# No ID range for Hoop

NEW_REFEREE_CLASS_ID = 3
NEW_REFEREE_ID_RANGE = range(13, 15)

BALL_CLASS_ID = 0

# Non-player objects configuration
NON_PLAYER_OBJECTS = {
    BALL_CLASS_ID: {'label': "Ball", 'color': (0, 255, 255)}, # Yellow
    NEW_HOOP_CLASS_ID: {'label': "Hoop", 'color': (255, 255, 0)}
}

# Video Video Processing
FPS = 30
PIXELS_PER_METER = 100  # Calibration factor

# Heatmap Config
HEATMAP_WIDTH = 940
HEATMAP_HEIGHT = 500
