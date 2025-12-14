class DrivingAgent:
    def __init__(self):
        self.state = "IDLE"

    def decide_action(self, detections, frame_width, frame_height):
        """
        Decides on an action based on detections and traffic light colors.
        
        Input: 
            detections: List of [class_id, conf, x1, y1, x2, y2, color_info]
            (Note: color_info is a new optional field we will add in app.py)
        """
        action = "ACCELERATE" 
        reason = "Path Clear"
        
        # --- ZONES CONFIGURATION ---
        center_x = frame_width // 2
        lane_width = frame_width * 0.45 
        danger_zone_left = center_x - (lane_width / 2)
        danger_zone_right = center_x + (lane_width / 2)
        
        # --- DISTANCE ESTIMATION (Heuristic based on Y-coordinate) ---
        # On a flat road, the lower an object is on screen, the closer it is.
        # Normalized Y (0.0 = top, 1.0 = bottom)
        WARNING_DISTANCE_Y = frame_height * 0.55  # ~15-20 meters away (Far)
        CRITICAL_DISTANCE_Y = frame_height * 0.80 # ~5 meters away (Very Close)

        for box in detections:
            # Unpack the new format (now includes color/extra info)
            if len(box) == 7:
                cls_id, conf, x1, y1, x2, y2, extra_info = box
            else:
                cls_id, conf, x1, y1, x2, y2 = box
                extra_info = None
            
            obj_center_x = (x1 + x2) / 2
            obj_bottom_y = y2  # The bottom of the box indicates how close it is
            
            # --- LOGIC 1: Traffic Lights (Class 9) ---
            if int(cls_id) == 9:
                if extra_info == "Red":
                    return "BRAKE", "Red Light Detected"
                elif extra_info == "Green":
                    # We don't return immediately; we keep checking for cars ahead
                    # But we assume the signal allows us to go.
                    pass 
                elif extra_info == "Yellow":
                    return "SLOW DOWN", "Yellow Light Detected"
                else:
                    # If uncertain/no color detected, assume caution but don't slam brakes
                    # unless it looks very close? For safety, let's slow down.
                    if obj_bottom_y < WARNING_DISTANCE_Y: 
                         # Light is far and we can't read it -> Caution
                         pass
                    else:
                         pass

            # --- LOGIC 2: Obstacle Avoidance (Cars, Pedestrians) ---
            obstacle_classes = [0, 1, 2, 3, 5, 7] # Person, Bike, Car, Motorbike, Bus, Truck
            
            if int(cls_id) in obstacle_classes:
                is_in_lane = (danger_zone_left < obj_center_x < danger_zone_right)
                
                if is_in_lane:
                    # DISTANCE CHECK
                    if obj_bottom_y > CRITICAL_DISTANCE_Y:
                        # Object is at the bottom of the screen (5 meters)
                        return "BRAKE", "Obstacle Critical (<5m)"
                    
                    elif obj_bottom_y > WARNING_DISTANCE_Y:
                        # Object is in the middle-field (10-15 meters)
                        # We change action to SLOW DOWN, but keep looking in case something else is closer
                        if action != "BRAKE": # Don't override a BRAKE decision
                            action = "SLOW DOWN"
                            reason = "Obstacle Ahead (15m)"
        
        return action, reason