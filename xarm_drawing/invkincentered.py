import math
import numpy as np
from scipy.spatial.transform import Rotation
from HersheyFonts import HersheyFonts
from xarm.wrapper import XArmAPI

text = 'THANK YOU'
arm_ip = '10.52.70.221'
height = 30
centx, centy = 300, 0
z_draw = 12
z_up = z_draw + 10
dist_to_paper = -5

draw_speed = 8
travel_speed = 10
test_gcode = True
air_test = True
pen_offset = None

a = [0, 200, 87, 0, 0, 0]
d = [243.3, 0, 0, 227.6, 0, 61.5]
alpha = np.radians([-90, 180, 90, 90, -90, 0])
shift = np.radians([0, -90, -90, 0, 0, 0])
rot = Rotation.from_euler('x', 180, degrees=True).as_matrix()

joint_low = np.radians([-360, -150, -3.5, -360, -124, -360])
joint_high = np.radians([360, 150, 300, 360, 124, 360])


def dh(theta, dist, length, twist):
    c, s = math.cos(theta), math.sin(theta)
    ca, sa = math.cos(twist), math.sin(twist)
    return np.array([
        [c, -s*ca, s*sa, length*c],
        [s, c*ca, -c*sa, length*s],
        [0, sa, ca, dist],
        [0, 0, 0, 1]
    ])


def forward(q):
    T = np.eye(4)
    frames = [T.copy()]
    for i in range(6):
        T = T @ dh(q[i] + shift[i], d[i], a[i], alpha[i])
        frames.append(T.copy())
    return T, frames


def jacobian(q, pen):
    T, frames = forward(q)
    tip = T[:3, 3] + T[:3, :3] @ pen
    J = np.zeros((6, 6))
    for i in range(6):
        axis = frames[i][:3, 2]
        origin = frames[i][:3, 3]
        J[:3, i] = np.cross(axis, tip - origin)
        J[3:, i] = axis
    return J


def inverse(point, q, pen):
    q = q.copy()
    for _ in range(300):
        T, _ = forward(q)
        tip = T[:3, 3] + T[:3, :3] @ pen
        pos_err = point - tip
        rot_err = Rotation.from_matrix(rot @ T[:3, :3].T).as_rotvec()

        if np.linalg.norm(pos_err) < 0.35 and np.linalg.norm(rot_err) < math.radians(0.5):
            return q

        J = jacobian(q, pen)
        J[3:, :] *= 80
        err = np.r_[pos_err, 80 * rot_err]
        damp = 12
        dq = J.T @ np.linalg.solve(J @ J.T + damp**2 * np.eye(6), err)

        biggest = np.max(np.abs(dq))
        if biggest > 0.12:
            dq *= 0.12 / biggest

        q += dq


font = HersheyFonts()
font.load_default_font()
font.normalize_rendering(height)

strokes = []
for start, end in font.lines_for_text(text):   
    start, end = tuple(start), tuple(end)
    if strokes and np.allclose(strokes[-1][-1], start):
        strokes[-1].append(end)
    else:
        strokes.append([start, end])

xs = [p[0] for stroke in strokes for p in stroke]
ys = [p[1] for stroke in strokes for p in stroke]
mid_x = (min(xs) + max(xs)) / 2
mid_y = (min(ys) + max(ys)) / 2


def to_robot(x, y):
    return centx + (y - mid_y), centy - (x - mid_x)


def move(x, y, z, f):
    return x, y, z + dist_to_paper + (20 if air_test else 0), f


gcode = []
prev = None
for stroke in strokes:
    pts = [to_robot(x, y) for x, y in stroke]
    sx, sy = pts[0]
    if prev is None or math.dist((sx, sy), prev) > 0.1:
        if prev is not None:
            gcode.append(move(prev[0], prev[1], z_up, travel_speed))
        gcode.append(move(sx, sy, z_up, travel_speed))
        gcode.append(move(sx, sy, z_draw, travel_speed))
    for x, y in pts:
        gcode.append(move(x, y, z_draw, draw_speed))
    prev = pts[-1]

if prev is not None:
    gcode.append(move(prev[0], prev[1], z_up, travel_speed))

path = []
for x, y, z, f in gcode:
    point = np.array([x, y, z], dtype=float)
    if not path:
        path.append((point, f))
        continue
    last = path[-1][0]
    steps = max(1, math.ceil(np.linalg.norm(point - last) / 2))
    for i in range(1, steps + 1):
        path.append((last + (point - last) * i / steps, f))

arm = None
started = False
try:
    if test_gcode:
        q = np.radians([0, 55, 70, 0, 10, 0])
        pen = np.array(pen_offset if pen_offset is not None else [0, 0, 0], dtype=float)
    else:
        pen = np.array(pen_offset, dtype=float)
        arm = XArmAPI(arm_ip)
        code, angles = arm.get_servo_angle(is_radian=False)
        q = np.radians(angles[:6])

        T, _ = forward(q)
        tip = T[:3, 3] + T[:3, :3] @ pen

        code, pose = arm.get_forward_kinematics(angles[:6], input_is_radian=False)
        if code != 0 or np.linalg.norm(np.array(pose[:3]) - T[:3, 3]) > 5:
            raise RuntimeError('DH model does not match the controller')

    moves = []
    for i, (point, f) in enumerate(path):
        new_q = inverse(point, q, pen)
        jump = np.max(np.abs(np.degrees(new_q - q)))
        moves.append((np.degrees(new_q).tolist(), f))
        q = new_q

    print(len(moves), 'points calculated')

    if test_gcode:
        print('Preview only')
    elif input('Type WRITE to start: ') == 'WRITE':
        code = arm.motion_enable(True)
        if code == 0:
            code = arm.set_mode(0)
        if code == 0:
            code = arm.set_state(0)
        if code != 0:
            raise RuntimeError(f'Arm setup failed: {code}')

        started = True
        for angles, f in moves:
            code = arm.set_servo_angle(angle=angles, speed=f, mvacc=40,
                                       is_radian=False, wait=True)
except BaseException:
    if started and arm is not None:
        arm.set_state(4)
    raise
finally:
    if arm is not None:
        arm.disconnect()