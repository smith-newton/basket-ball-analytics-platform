
import numpy as np
import torch
from collections import defaultdict, deque
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
from torchvision import transforms
import cv2

try:
    import torchreid
    HAS_TORCHREID = True
except ImportError:
    HAS_TORCHREID = False
    print("Warning: torchreid not found. Re-ID features will not work.")

from .config import (
    TEAM_A, TEAM_B, NEW_REFEREE_CLASS_ID, NEW_REFEREE_ID_RANGE,
    NEW_COACH_CLASS_ID, NEW_COACH_ID_RANGE
)

class StablePlayerTracker:
    def __init__(self, reid_model=None, device='cpu'):
        self.feature_db = defaultdict(lambda: deque(maxlen=5))
        self.assigned_ids = {}
        self.player_history = defaultdict(lambda: deque(maxlen=30))
        self.position_data = defaultdict(list)
        self.speed_data = defaultdict(list)
        self.hoop_positions = []

        # Available IDs based on configuration
        self.available_ids = {
            TEAM_A['class_id']: deque(TEAM_A['id_range']),
            TEAM_B['class_id']: deque(TEAM_B['id_range']),
            NEW_REFEREE_CLASS_ID: deque(NEW_REFEREE_ID_RANGE),
            NEW_COACH_CLASS_ID: deque(NEW_COACH_ID_RANGE)
        }
        self.last_seen = {}
        self.current_frame = 0

        self.similarity_threshold = 0.85
        self.max_frames_missing = 90
        self.min_features = 3
        
        self.reid_model = reid_model
        self.device = device
        
        # Transformation for ReID
        self.transform = transforms.Compose([
            transforms.Resize((256, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def extract_features(self, image):
        if not HAS_TORCHREID or self.reid_model is None:
            return np.zeros(512) # dummy feature
            
        if image is None or image.size == 0:
            return None

        # Ensure image is RGB (OpenCV uses BGR)
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            features = self.reid_model(tensor)
        return features.cpu().numpy()[0]

    def get_avg_feature(self, static_id, cls):
        features = list(self.feature_db[(cls, static_id)])
        if len(features) >= self.min_features:
            return np.mean(features, axis=0)
        elif features:
            return features[-1]
        return None

    def assign_id(self, track_id, cls, crop):
        self.current_frame += 1

        if track_id in self.assigned_ids:
            static_id = self.assigned_ids[track_id]
            self.last_seen[static_id] = self.current_frame

            feature = self.extract_features(crop)
            if feature is not None:
                self.feature_db[(cls, static_id)].append(feature)
            return static_id

        feature = self.extract_features(crop)
        if feature is None:
            return None

        best_match = None
        max_sim = -1

        for static_id, last_frame in list(self.last_seen.items()):
            is_valid_class_match = False
            # Check if class matches the static_id's group
            if static_id in TEAM_A['id_range'] and cls == TEAM_A['class_id']:
                is_valid_class_match = True
            elif static_id in TEAM_B['id_range'] and cls == TEAM_B['class_id']:
                is_valid_class_match = True
            elif static_id in NEW_REFEREE_ID_RANGE and cls == NEW_REFEREE_CLASS_ID:
                is_valid_class_match = True
            elif static_id in NEW_COACH_ID_RANGE and cls == NEW_COACH_CLASS_ID:
                is_valid_class_match = True

            if not is_valid_class_match:
                continue

            if self.current_frame - last_frame > self.max_frames_missing:
                continue

            avg_feature = self.get_avg_feature(static_id, cls)
            if avg_feature is not None:
                sim = cosine_similarity([feature], [avg_feature])[0][0]
                if sim > max_sim and sim > self.similarity_threshold:
                    max_sim = sim
                    best_match = static_id

        if best_match:
            self.assigned_ids[track_id] = best_match
            self.last_seen[best_match] = self.current_frame
            self.feature_db[(cls, best_match)].append(feature)
            return best_match

        # Assign new ID from available pool
        if cls in self.available_ids and self.available_ids[cls]:
            new_id = self.available_ids[cls].popleft()
            self.assigned_ids[track_id] = new_id
            self.last_seen[new_id] = self.current_frame
            self.feature_db[(cls, new_id)].append(feature)
            return new_id

        return None # No ID available

    def cleanup(self):
        recycled_ids = defaultdict(list)

        for static_id, last_frame in list(self.last_seen.items()):
            if self.current_frame - last_frame > self.max_frames_missing:
                if static_id in TEAM_A['id_range']:
                    recycled_ids[TEAM_A['class_id']].append(static_id)
                elif static_id in TEAM_B['id_range']:
                    recycled_ids[TEAM_B['class_id']].append(static_id)
                elif static_id in NEW_REFEREE_ID_RANGE:
                    recycled_ids[NEW_REFEREE_CLASS_ID].append(static_id)
                elif static_id in NEW_COACH_ID_RANGE:
                    recycled_ids[NEW_COACH_CLASS_ID].append(static_id)

                del self.last_seen[static_id]
                self.feature_db.pop((self._get_class_from_static_id(static_id), static_id), None)

        for cls, ids_to_recycle in recycled_ids.items():
            for static_id in ids_to_recycle:
                if static_id not in self.available_ids[cls]:
                    self.available_ids[cls].append(static_id)

        for track_id, static_id in list(self.assigned_ids.items()):
            if static_id not in self.last_seen:
                del self.assigned_ids[track_id]

    def _get_class_from_static_id(self, static_id):
        if static_id in TEAM_A['id_range']:
            return TEAM_A['class_id']
        elif static_id in TEAM_B['id_range']:
            return TEAM_B['class_id']
        elif static_id in NEW_REFEREE_ID_RANGE:
            return NEW_REFEREE_CLASS_ID
        elif static_id in NEW_COACH_ID_RANGE:
            return NEW_COACH_CLASS_ID
        return None

    def record_speed(self, static_id, speed):
        self.speed_data[static_id].append((self.current_frame, speed))

    def record_position(self, static_id, position):
        self.position_data[static_id].append(position)

    def record_hoop_position(self, position):
        if position not in self.hoop_positions:
            self.hoop_positions.append(position)
            
    def get_all_speed_data(self):
        return self.speed_data

