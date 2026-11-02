import socket
import time
import math
from HersheyFonts import HersheyFonts


from xarm.wrapper import XArmAPI
import time

# Initialize connection to your Lite 6 (replace with your robot's IP address)
arm = XArmAPI("10.52.70.211")

# Switch to Manual Mode (Mode 2) and enable state (State 0)
arm.set_mode(0)
arm.set_state(0)

# The robot is now in zero-gravity manual/drag mode
print("Manual mode enabled. You can drag the robot by hand.")

# To switch back to normal position control mode (Mode 0) later:
# arm.set_mode(0)
# arm.set_state(0)

text = "THANK YOU" # change text here
arm_ip = "10.52.70.211"
arm_port = 504   # uFactory G-code TCP port (was 18333 — try that if 504 refuses)
height = 30
centx, centy = 300, 0
z_draw = 12
z_up = z_draw + 10
dist_to_paper = -5 # 0 means its actually writing :>
draw_speed = 1000     # mm/min, pen on the paper (raise to draw faster)
travel_speed = 9000   # mm/min, pen in the air between strokes (can be much faster)
test_gcode = False
line_ending = "\n"   # uFactory G-code is newline-terminated

font = HersheyFonts()
font.load_default_font('futural') 
font.normalize_rendering(height)
strokes = [i for i in font.strokes_for_text(text)]

xs = [x for s in strokes for x, y in s]
ys = [y for s in strokes for x, y, in s]
mid_x = (min(xs) + max(xs))/2
mid_y = (min(ys) + max(ys))/2

def to_robot(x,y):
    return centx + (y-mid_y), centy - (x- mid_x)
def move(x,y,z,f):
    return f"G1 X{x:.2f} Y{y:.2f} Z{z + dist_to_paper:.2f} A180 B0 C0 F{f}"

gcode = []
prev = None   # where the pen currently rests (pen-down), or None if lifted
for stroke in strokes:
    pts = [to_robot(x,y) for x,y in stroke]
    sx, sy = pts[0]
    # Only lift/travel/plunge if this stroke doesn't continue from the last one
    if prev is None or abs(sx - prev[0]) > 0.1 or abs(sy - prev[1]) > 0.1:
        if prev is not None:
            gcode.append(move(prev[0], prev[1], z_up, travel_speed))  # lift where we stopped
        gcode.append(move(sx, sy, z_up,   travel_speed))   # travel above the new start
        gcode.append(move(sx, sy, z_draw, travel_speed))   # plunge straight down onto paper
    for x, y in pts:
        gcode.append(move(x, y, z_draw, draw_speed))       # draw, pen stays down
    prev = pts[-1]
if prev is not None:
    gcode.append(move(prev[0], prev[1], z_up, travel_speed))  # final lift at the very end


sock = socket.create_connection((arm_ip, arm_port), timeout=5)
sock.settimeout(0.5)   # this arm may not ack; don't hang waiting for a reply

def send(line):
    sock.sendall((line+line_ending).encode())
    try:
        resp = sock.recv(1024)
    except TimeoutError:
        resp = None        # no reply within 0.5s — maybe this arm doesn't ack
    if resp == b"":
        raise ConnectionError(f"Arm closed the connection while sending {line!r}.")
    print(line, "->", repr(resp))
    return resp

for line in ["M100 P1", "M104 P0"]:   # enable all joints, then set state READY
    send(line)
for line in gcode:
    send(line)

input("Press Enter when done")
sock.close()
thefont = HersheyFonts()
print(thefont.default_font_names)
