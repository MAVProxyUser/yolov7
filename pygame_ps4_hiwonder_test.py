from pyPS4Controller.controller import Controller
import HiwonderServoController as servo
import os
import time
import pygame

# Setup the HiWonder Servo
servo.setConfig('/dev/ttyUSB0', 1)
g = [1, 5]
tilt, pan = g

# Setup GPIO
os.system('echo 20 > /sys/class/gpio/export')
os.system('echo out > /sys/class/gpio/gpio20/direction')

# Query and Display current servo position
boot_pos = servo.multServoPosRead(g)
print("Boot Up Servo Position: Tilt/Pan")
print(boot_pos[1], boot_pos[5])

pan_max = 750
pan_min = 300
tilt_max = 650
tilt_min = 500
home_pan = 530
home_tilt = 500

# Set home position of servos
def xy_home():

#    servo.moveServo(tilt, tilt_min + (tilt_max - tilt_min)//2, 500)
#    servo.moveServo(pan, pan_min + (pan_max - pan_min)//2, 500)
#    servo.moveServo(tilt, tilt_min + (tilt_max - tilt_min), 500)
#    servo.moveServo(pan, pan_min + (pan_max - pan_min), 500)
    servo.moveServo(tilt, home_pan, 500)
    servo.moveServo(pan, home_tilt, 500)

# Test Pan Sweep
def test_pan():
    for i in range(300, 700):
         servo.moveServo(pan, i, 500)

# Test Tilt
def test_tilt():
    for i in range(400, 500):
         servo.moveServo(tilt, i, 500)

def start_fire():
    print("start firing")
    os.system('echo 1 > /sys/class/gpio/gpio20/value')

def stop_fire():
    print("stop firing")
    os.system('echo 0 > /sys/class/gpio/gpio20/value')

xy_home()
test_pan()
xy_home()

class PS4Controller(object):
    """Class representing the PS4 controller. Pretty straightforward functionality."""

    controller = None
    axis_data = None
    button_data = None
    hat_data = None
    current_pan = home_pan
    current_tilt = home_tilt
    pan_increment = 5  # Increase pan increment
    tilt_increment = 2.5  # Increase tilt increment
    deadzone = 0.2  # Define deadzone

    def __init__(self):
        """Initialize the joystick components"""
        pygame.init()
        pygame.joystick.init()
        self.controller = pygame.joystick.Joystick(0)
        self.controller.init()

    def listen(self):
        """Listen for events to happen"""

        if not self.axis_data:
            self.axis_data = {}

        if not self.button_data:
            self.button_data = {}
            for i in range(self.controller.get_numbuttons()):
                self.button_data[i] = False

        if not self.hat_data:
            self.hat_data = {}
            for i in range(self.controller.get_numhats()):
                self.hat_data[i] = (0, 0)

        while True:
            for event in pygame.event.get():
                if event.type == pygame.JOYAXISMOTION:
                    self.axis_data[event.axis] = round(event.value, 2)

                    if event.axis == 2:
                        if abs(event.value) < self.deadzone:  # Within deadzone
                            return
                        else:
                            target_pan = self.current_pan - int(event.value * self.pan_increment)
                            target_pan = min(max(target_pan, pan_min), pan_max)
                            pan_diff = abs(target_pan - self.current_pan)
                            pan_dir = 1 if target_pan > self.current_pan else -1
                            for i in range(pan_diff):
                                self.current_pan += pan_dir
                                servo.moveServo(pan, self.current_pan, 1)
                                time.sleep(0.01)

                    elif event.axis == 5:
                        if abs(event.value) < self.deadzone:  # Within deadzone
                            return
                        else:
                            target_tilt = self.current_tilt + int(event.value * self.tilt_increment)
                            target_tilt = min(max(target_tilt, tilt_min), tilt_max)
                            tilt_diff = abs(target_tilt - self.current_tilt)
                            tilt_dir = 1 if target_tilt > self.current_tilt else -1
                            for i in range(tilt_diff):
                                self.current_tilt += tilt_dir
                                servo.moveServo(tilt, self.current_tilt, 1)
                                time.sleep(0.01)

                elif event.type == pygame.JOYBUTTONDOWN:
                    self.button_data[event.button] = True
                    if event.button == 7:
                        print("FIRE")
                        start_fire()
                elif event.type == pygame.JOYBUTTONUP:
                    self.button_data[event.button] = False
                    if event.button == 7:
                        print("STOP FIRE")
                        stop_fire()
                elif event.type == pygame.JOYHATMOTION:
                    self.hat_data[event.hat] = event.value
                    if event.hat == 0:
                        if event.value == (1, 0):
                            print("right")
                            self.current_pan -= self.pan_increment
                            self.current_pan = min(self.current_pan, pan_max)
                            servo.moveServo(pan, self.current_pan, 500)
                        elif event.value == (-1, 0):
                            print("left")
                            self.current_pan += self.pan_increment
                            self.current_pan = max(self.current_pan, pan_min)
                            servo.moveServo(pan, self.current_pan, 500)
                        elif event.value == (0, 1):
                            print("up")
                            self.current_tilt -= self.tilt_increment
                            self.current_tilt = min(self.current_tilt, tilt_max)
                            servo.moveServo(tilt, self.current_tilt, 500)
                        elif event.value == (0, -1):
                            print("down")
                            self.current_tilt += self.tilt_increment
                            self.current_tilt = max(self.current_tilt, tilt_min)
                            servo.moveServo(tilt, self.current_tilt, 500)

if __name__ == "__main__":

    ps4 = PS4Controller()
    while True:
        ps4.listen()
