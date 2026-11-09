"""
hand_drive.py - drive a VEX AIM robot with your hand in the webcam.
 
MODE = "signs" (default): hand signs control the robot (see SIGN_TO_COMMAND below)
MODE = "position":        hand left / middle / right of the screen steers,
                          open palm stops
Either mode: no hand for half a second -> stop
 
Put this file and gesture_recognizer.task in the AIM_Websocket_Library folder.
 
Run (venv active):
    python hand_drive.py --no-robot     # test camera + hand tracking only
    python hand_drive.py                # drive the robot (join the robot's Wi-Fi first)
 
Keys in the video window:  SPACE = emergency stop,  Q = quit
"""
 
import argparse
import time
 
import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision
 
# ---------------- settings to tune ----------------
MODE = "signs"            # "signs" = hand signs below, "position" = hand left/middle/right steers
 
# Hand sign -> robot command (used when MODE = "signs"). Change these to match your doc.
# Options: "FORWARD", "REVERSE", "LEFT", "RIGHT", "STOP", or None (ignore that sign)
SIGN_TO_COMMAND = {
    "Open_Palm":   "STOP",
    "Closed_Fist": "FORWARD",
    "Thumb_Down":  "REVERSE",
    "Victory":     "LEFT",     # peace sign
    "ILoveYou":    "RIGHT",    # thumb + index + pinky
    "Thumb_Up":    None,
    "Pointing_Up": None,
}
MIN_CONFIDENCE = 0.6      # ignore sign guesses below this
 
LEFT_EDGE = 0.35         # hand x below this  -> LEFT
RIGHT_EDGE = 0.65         # hand x above this  -> RIGHT  (between = FORWARD)
DRIVE_SPEED = 40          # percent, 0-100 (start slow)
TURN_SPEED = 30           # percent, 0-100
STABLE_FRAMES = 4         # command must repeat this many frames before switching
NO_HAND_STOP_SEC = 0.5    # stop if no hand seen for this long
OPEN_PALM_STOPS = True    # open palm = stop
MODEL_PATH = "gesture_recognizer.task"
# --------------------------------------------------
 
 
class RobotLink:
    """Sends commands to the robot, or just prints them with --no-robot."""
 
    def __init__(self, use_robot, host):
        self.robot = None
        if use_robot:
            from vex import Robot, TurnType, Color, LightType
            self.TurnType, self.Color, self.LightType = TurnType, Color, LightType
            self.robot = Robot(host) if host else Robot()
 
    def send(self, command):
        print("->", command)
        r = self.robot
        if r is None:
            return
        if command == "FORWARD":
            r.move_at(0, DRIVE_SPEED)
        elif command == "REVERSE":
            r.move_at(180, DRIVE_SPEED)
        elif command == "LEFT":
            r.turn(self.TurnType.LEFT, TURN_SPEED)
        elif command == "RIGHT":
            r.turn(self.TurnType.RIGHT, TURN_SPEED)
        else:
            r.stop_all_movement()
        r.led.on(self.LightType.ALL_LEDS,
                 self.Color.RED if command == "STOP" else self.Color.GREEN)
 
 
def make_recognizer():
    options = vision.GestureRecognizerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
    )
    return vision.GestureRecognizer.create_from_options(options)
 
 
def read_hand(result):
    """Return (x, y, gesture_name) for the first hand, or None if no hand."""
    if not result.hand_landmarks:
        return None
    palm = result.hand_landmarks[0][9]  # landmark 9 = middle of the palm
    gesture = "None"
    if result.gestures:
        top = result.gestures[0][0]
        if top.score >= MIN_CONFIDENCE:
            gesture = top.category_name
    return palm.x, palm.y, gesture
 
 
def pick_command(hand, last_hand_time):
    """Decide what the robot should do. Returns (command, status text)."""
    if hand is None:
        if time.time() - last_hand_time > NO_HAND_STOP_SEC:
            return "STOP", "Can't see your hand - robot stopped"
        return None, "Looking for hand..."  # brief dropout: keep current command
    x, _, gesture = hand
    if MODE == "signs":
        command = SIGN_TO_COMMAND.get(gesture)
        if command is None:
            return None, f"sign: {gesture} (no action)"  # keep doing the last command
        return command, f"sign: {gesture}"
    # MODE == "position"
    if OPEN_PALM_STOPS and gesture == "Open_Palm":
        return "STOP", "sign: Open_Palm"
    if x < LEFT_EDGE:
        return "LEFT", f"sign: {gesture}"
    if x > RIGHT_EDGE:
        return "RIGHT", f"sign: {gesture}"
    return "FORWARD", f"sign: {gesture}"
 
 
def draw(frame, hand, command, status):
    h, w = frame.shape[:2]
    if MODE == "position":
        lx, rx = int(w * LEFT_EDGE), int(w * RIGHT_EDGE)
        cv2.line(frame, (lx, 0), (lx, h), (255, 255, 255), 2)
        cv2.line(frame, (rx, 0), (rx, h), (255, 255, 255), 2)
        for label, cx in (("LEFT", lx // 2), ("FORWARD", (lx + rx) // 2), ("RIGHT", (rx + w) // 2)):
            size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            cv2.putText(frame, label, (cx - size[0] // 2, h - 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    else:
        # cheat sheet of signs down the right side
        y = 90
        for sign, cmd in SIGN_TO_COMMAND.items():
            if cmd:
                cv2.putText(frame, f"{sign} = {cmd}", (w - 300, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                y += 28
    if hand is not None:
        cv2.circle(frame, (int(hand[0] * w), int(hand[1] * h)), 14, (0, 255, 0), -1)
    cv2.rectangle(frame, (0, 0), (w, 60), (0, 0, 0), -1)
    color = (0, 0, 255) if command == "STOP" else (0, 255, 0)
    cv2.putText(frame, f"ROBOT: {command}", (16, 42),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
    cv2.putText(frame, status, (16, h - 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
 
 
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-robot", action="store_true", help="test without the robot")
    parser.add_argument("--host", default="", help="robot IP (default 192.168.4.1)")
    parser.add_argument("--camera", type=int, default=0, help="webcam number")
    args = parser.parse_args()
 
    link = RobotLink(not args.no_robot, args.host)
    recognizer = make_recognizer()
    cam = cv2.VideoCapture(args.camera)
    if not cam.isOpened():
        raise SystemExit("Could not open webcam. Try --camera 1, and allow camera access "
                         "for Terminal in System Settings > Privacy & Security > Camera.")
 
    current_cmd = "STOP"
    candidate, candidate_count = None, 0
    last_hand_time = time.time()
    last_ts = 0
    link.send(current_cmd)
 
    try:
        while True:
            ok, frame = cam.read()
            if not ok:  # Mac cameras can give empty frames while warming up
                time.sleep(0.05)
                continue
            frame = cv2.flip(frame, 1)  # mirror: your left = screen left
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            ts = max(int(time.time() * 1000), last_ts + 1)  # must always increase
            last_ts = ts
            result = recognizer.recognize_for_video(image, ts)
 
            hand = read_hand(result)
            if hand is not None:
                last_hand_time = time.time()
            wanted, status = pick_command(hand, last_hand_time)
 
            if wanted is not None:
                if wanted == candidate:
                    candidate_count += 1
                else:
                    candidate, candidate_count = wanted, 1
                # STOP happens right away; other commands wait to be steady
                ready = wanted == "STOP" or candidate_count >= STABLE_FRAMES
                if ready and wanted != current_cmd:
                    current_cmd = wanted
                    link.send(current_cmd)
 
            draw(frame, hand, current_cmd, status)
            cv2.imshow("GestureBot  (SPACE = stop, Q = quit)", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord(" "):
                current_cmd = "STOP"
                link.send(current_cmd)
            elif key == ord("q"):
                break
    finally:
        link.send("STOP")
        cam.release()
        cv2.destroyAllWindows()
 
 
if __name__ == "__main__":
    main()
 